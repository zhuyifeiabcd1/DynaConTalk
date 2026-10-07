"""Transcript and word timings of a job's audio (Qwen3-ASR + Qwen3-ForcedAligner).

Writes `<job>/transcripts.json` (text per chunk) and `<job>/features/words.json`::

    {"fps": 30, "language": "English",
     "words": [{"i": 0, "text": "Holistic", "start": 0.42, "end": 0.81, "sf": 13, "ef": 24, "chunk": 0, "sentence": 0}, ...],
     "sentences": [{"i": 0, "text": "...", "start": 0.42, "end": 3.9, "sf": 13, "ef": 117, "chunk": 0}, ...]}

The generation job calls `run_asr` and `build_words_json`; run as a module it is the
"transcribe" job of the Studio (word timings for older jobs):

    python -m webui.transcribe --job-dir <transcribe job> --request-json <request.json> [--status-json ...]
"""
import argparse
import os
import re
import time
from pathlib import Path

import numpy as np

from webui.pipeline import FPS, Status, read_json, report_failure, write_json

ASR_MODEL = os.environ.get("STUDIO_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B")
ALIGNER_MODEL = os.environ.get("STUDIO_ALIGNER_MODEL", "Qwen/Qwen3-ForcedAligner-0.6B")
MAX_ALIGN_SECONDS = 60.0
SENTENCE_END = re.compile(r"[.!?。！？;；]+[\"'”’)]*$")
_NORM = re.compile(r"[^0-9a-z一-鿿]+")


def _norm(token: str) -> str:
    return _NORM.sub("", token.lower())


def pick_device(min_free_mb: int = 6000) -> str:
    """The GPU if it has min_free_mb free, otherwise the CPU (ASR is small enough for either)."""
    try:
        import torch

        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            if free / 1e6 >= min_free_mb:
                return "cuda:0"
            print(f"[device] only {free / 1e9:.1f} GB free on the GPU; using the CPU")
    except Exception as exc:  # noqa: BLE001
        print(f"[device] cuda probe failed: {exc}")
    return "cpu"


def load_wav16k(path: Path) -> np.ndarray:
    import soundfile as sf

    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    return wav.astype(np.float32)


def run_asr(audio: np.ndarray, chunks: list[dict], device: str, language: str) -> list[dict]:
    """Text per chunk ({"index", "frame_start", "frame_end"}); audio cut in pieces of at most 30 s."""
    import torch
    from qwen_asr import Qwen3ASRModel

    print(f"[asr] loading {ASR_MODEL} on {device}")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    asr = Qwen3ASRModel.from_pretrained(ASR_MODEL, dtype=dtype, device_map=device, max_inference_batch_size=1,
                                        max_new_tokens=512)
    out = []
    max_seg = 30 * 16000
    for row in chunks:
        a, b = int(round(row["frame_start"] / FPS * 16000)), int(round(row["frame_end"] / FPS * 16000))
        wav = audio[a:b]
        pieces = [wav[i:i + max_seg] for i in range(0, wav.shape[0], max_seg)]
        pieces = [p for p in pieces if p.shape[0] >= 8000]
        text, lang = "", ""
        if pieces:
            try:
                results = asr.transcribe([(p, 16000) for p in pieces], language=language)
                text = " ".join(r.text.strip() for r in results if r.text and r.text.strip()).strip()
                lang = ",".join(sorted({r.language for r in results if r.language}))
            except Exception as exc:  # noqa: BLE001
                print(f"[asr] chunk {row['index']} failed: {exc}")
        out.append({"index": int(row["index"]), "frame_start": int(row["frame_start"]),
                    "frame_end": int(row["frame_end"]), "language": lang, "text": text})
    del asr
    return out


def _split_text_for_pieces(text: str, piece_lengths: list[int]) -> list[str]:
    """Distribute the words over pieces in proportion to their duration (chunks over MAX_ALIGN_SECONDS)."""
    toks = text.split()
    total = float(sum(piece_lengths)) or 1.0
    bounds = np.cumsum([int(round(len(toks) * (n / total))) for n in piece_lengths])
    bounds[-1] = len(toks)
    pieces, prev = [], 0
    for b in bounds:
        pieces.append(" ".join(toks[prev:b]))
        prev = int(b)
    return pieces


def align_words(audio: np.ndarray, chunks: list[dict], device: str, language: str, status=None) -> list[dict]:
    import torch
    from qwen_asr import Qwen3ForcedAligner

    print(f"[align] loading {ALIGNER_MODEL} on {device}")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    aligner = Qwen3ForcedAligner.from_pretrained(ALIGNER_MODEL, dtype=dtype, device_map=device)
    words: list[dict] = []
    for ci, row in enumerate(chunks):
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        if status is not None:
            status("running", "align", 0.3 + 0.6 * ci / max(1, len(chunks)), f"Aligning chunk {ci + 1}/{len(chunks)}")
        a, b = int(round(row["frame_start"] / FPS * 16000)), int(round(row["frame_end"] / FPS * 16000))
        wav = audio[a:b]
        lang = (row.get("language") or language or "English").split(",")[0] or "English"
        max_len = int(MAX_ALIGN_SECONDS * 16000)
        if wav.shape[0] <= max_len:
            pieces, texts = [(0, wav)], [text]
        else:
            pieces = [(s, wav[s:s + max_len]) for s in range(0, wav.shape[0], max_len)]
            texts = _split_text_for_pieces(text, [p.shape[0] for _, p in pieces])
        for (offset, piece), piece_text in zip(pieces, texts):
            if not piece_text.strip() or piece.shape[0] < 4000:
                continue
            try:
                result = aligner.align(audio=(piece, 16000), text=piece_text, language=lang)[0]
            except Exception as exc:  # noqa: BLE001
                print(f"[align] chunk {ci} piece @{offset} failed: {exc}")
                continue
            base = row["frame_start"] / FPS + offset / 16000.0
            for it in result.items:
                st = float(getattr(it, "start_time", 0.0) or 0.0)
                en = float(getattr(it, "end_time", st) or st)
                if en > 500:  # milliseconds
                    st, en = st / 1000.0, en / 1000.0
                s_abs, e_abs = base + st, base + max(en, st + 1.0 / FPS)
                words.append({
                    "i": len(words), "text": str(getattr(it, "text", "")).strip(), "start": round(s_abs, 3),
                    "end": round(e_abs, 3), "sf": int(round(s_abs * FPS)),
                    "ef": max(int(round(s_abs * FPS)) + 1, int(round(e_abs * FPS))), "chunk": int(row["index"]),
                })
    del aligner
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return [w for w in words if w["text"]]


def attach_punctuation(words: list[dict], rows: list[dict]) -> None:
    """Give aligned words back their punctuated form from the transcript (captions, sentence breaks).

    The aligner strips punctuation and hyphens ("co-speech," -> "cospeech"); the transcript tokens
    are walked in order and matched on normalized forms, tolerating a few skipped tokens.
    """
    by_chunk: dict[int, list[dict]] = {}
    for w in words:
        by_chunk.setdefault(int(w["chunk"]), []).append(w)
    for row in rows:
        tokens = str(row.get("text") or "").split()
        ti = 0
        for w in by_chunk.get(int(row["index"]), []):
            target = _norm(w["text"])
            hit = None
            for look in range(ti, min(len(tokens), ti + 4)):
                if _norm(tokens[look]) == target:
                    hit = look
                    break
            if hit is None:
                # the aligner sometimes merges two short tokens
                for look in range(ti, min(len(tokens) - 1, ti + 3)):
                    if _norm(tokens[look] + tokens[look + 1]) == target:
                        hit = look + 1
                        w["text"] = tokens[look] + " " + tokens[look + 1]
                        break
                if hit is None:
                    continue
            else:
                w["text"] = tokens[hit]
            ti = hit + 1


def split_sentences(words: list[dict], max_words: int = 24) -> list[dict]:
    """Sentences end at final punctuation, at chunk borders, or after max_words words."""
    groups, cur = [], []
    for idx, w in enumerate(words):
        cur.append(w)
        last_of_chunk = idx + 1 < len(words) and words[idx + 1]["chunk"] != w["chunk"]
        if SENTENCE_END.search(w["text"]) or last_of_chunk or len(cur) >= max_words:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    out = []
    for i, group in enumerate(groups):
        for w in group:
            w["sentence"] = i
        out.append({"i": i, "text": " ".join(w["text"] for w in group), "start": group[0]["start"],
                    "end": group[-1]["end"], "sf": group[0]["sf"], "ef": group[-1]["ef"], "chunk": group[0]["chunk"]})
    return out


def chunk_table(job_dir: Path, total_frames: int) -> list[dict]:
    rows = read_json(job_dir / "manifest.json").get("chunks") or []
    if rows:
        return [{"index": int(r["index"]), "frame_start": int(r["frame_start"]), "frame_end": int(r["frame_end"])}
                for r in rows]
    return [{"index": 0, "frame_start": 0, "frame_end": total_frames}]


def build_words_json(job_dir: Path, device: str, language: str, force_asr: bool, status=None) -> Path:
    """Word timings of a generation job; runs ASR first when transcripts.json is missing or force_asr."""
    audio_path = sorted((job_dir / "audio").glob("*_16k.wav"))[0]
    if status is not None:
        status("running", "audio", 0.05, f"Loading {audio_path.name}")
    audio = load_wav16k(audio_path)
    total_frames = int(len(audio) / 16000.0 * FPS)
    chunks = chunk_table(job_dir, total_frames)
    transcripts_path = job_dir / "transcripts.json"
    texts = {}
    if transcripts_path.exists() and not force_asr:
        texts = {int(r["index"]): r for r in read_json(transcripts_path).get("chunks") or []}
    if not texts or any(int(c["index"]) not in texts for c in chunks):
        if status is not None:
            status("running", "asr", 0.1, "Running speech recognition")
        rows = run_asr(audio, chunks, device, language)
        texts = {int(r["index"]): r for r in rows}
        if not transcripts_path.exists() or force_asr:
            write_json(transcripts_path, {"chunks": rows})
    rows = [{**c, "text": texts[int(c["index"])].get("text", ""), "language": texts[int(c["index"])].get("language", "")}
            for c in chunks]
    words = align_words(audio, rows, device, language, status)
    attach_punctuation(words, rows)
    sentences = split_sentences(words)
    out = job_dir / "features" / "words.json"
    write_json(out, {
        "fps": FPS, "total_frames": total_frames, "language": language, "source": ALIGNER_MODEL, "device": device,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "words": words, "sentences": sentences,
        "chunks": [{"index": r["index"], "frame_start": r["frame_start"], "frame_end": r["frame_end"], "text": r["text"]}
                   for r in rows],
    })
    print(f"[words] wrote {out}: {len(words)} words, {len(sentences)} sentences")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Word timings for a Studio generation job.")
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--status-json")
    parser.add_argument("--artifacts-json")
    args = parser.parse_args()
    status = Status(args.status_json)
    try:
        req = read_json(Path(args.request_json))
        target = Path(req["target_job_dir"]).resolve()
        device = pick_device()
        out = build_words_json(target, device, req.get("language") or "English", bool(req.get("force_asr")), status)
        if args.artifacts_json:
            write_json(Path(args.artifacts_json), {"job_id": Path(args.job_dir).name, "target_job_dir": str(target),
                                                   "items": [{"name": "words.json", "path": str(out), "kind": "json"}]})
        status("running", "done", 0.99, "Transcript aligned")
    except Exception as exc:
        report_failure(args.status_json, exc)
        raise


if __name__ == "__main__":
    main()
