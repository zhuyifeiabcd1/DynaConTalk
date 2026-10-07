"""Generation job: speech audio -> body + face motion, chunk by chunk.

    python -m webui.generate --job-dir <job> --request-json <job>/request.json [--status-json ...] [--artifacts-json ...]

request.json: {"audio": {"path"}, "identity", "trajectory", "traj_script", "guidance_scale", "skip_asr",
"language", "render"}.

The audio is cut into chunks as long as the chosen trajectory (the trajectory restarts in every
chunk). Each chunk is generated window by window by the editable body model (speech, root
trajectory, body shape, speaker) and the face model (speech, body shape, speaker); both start
from the first frames of a fixed BEAT2 test sequence (assets/seed.npz). The rendered root
translation is predicted from the generated pose (webui/trajectory.py).
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from webui import traj_script as ts
from webui.pipeline import (
    ASSETS_DIR, BODY_CHECKPOINT, EXPR_DIM, FACE_CHECKPOINT, FPS, POSE_DIM, TRAJECTORY_CHECKPOINT, Generator, Status,
    chunk_name, concatenate_chunks, free_cuda, read_json, report_failure, setup_torch, write_json,
)
from webui.rotations import rot6d_to_aa


def load_identity(identity_id: str) -> dict:
    for line in (ASSETS_DIR / "identities" / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip() and json.loads(line).get("id") == identity_id:
            record = json.loads(line)
            return {"activity": np.asarray(record["activity"], dtype=np.float32).reshape(-1),
                    "shape_betas": np.asarray(record["shape_betas"], dtype=np.float32).reshape(-1)}
    raise KeyError(f"identity {identity_id!r} not found under {ASSETS_DIR / 'identities'}")


def load_trajectory(trajectory_id: str) -> dict:
    """Root translation [T, 3] and root orientation [T, 3] (axis-angle) of a trajectory asset."""
    path = ASSETS_DIR / "trajectories" / f"{trajectory_id}.npz"
    if not path.exists():
        raise KeyError(f"trajectory {trajectory_id!r} not found under {ASSETS_DIR / 'trajectories'}")
    with np.load(path) as z:
        return {"trans": np.asarray(z["trans"], dtype=np.float32), "root_orient": np.asarray(z["root_orient"], dtype=np.float32)}


def convert_audio(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", str(dst)], check=True)


def speech_features(audio: np.ndarray, path: Path, device: str) -> dict:
    if path.exists():  # rerun of a job
        with np.load(path) as z:
            return {k: z[k] for k in ("rhythm", "mel", "semantic")}
    from src.data.speech_features import extract_hubert, extract_mel, extract_rhythm, load_hubert

    print("[features] rhythm / mel")
    rhythm = extract_rhythm(audio)
    mel = extract_mel(audio)
    print("[features] HuBERT")
    processor, hubert = load_hubert(device)
    semantic = extract_hubert(processor, hubert, audio, device)
    del processor, hubert
    free_cuda()
    n = min(rhythm.shape[0], mel.shape[0], semantic.shape[0])
    features = {"rhythm": rhythm[:n].astype(np.float32), "mel": mel[:n].astype(np.float32),
                "semantic": semantic[:n].astype(np.float32)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **features)
    return features


def clip_texts(texts: list[str], fallback: np.ndarray, device: str) -> np.ndarray:
    """CLIP embedding per chunk transcript; chunks without speech use the seed sequence's transcript."""
    from src.data.speech_features import encode_text, load_clip

    model, clip_module = load_clip(device)
    out = [fallback.astype(np.float32) if not t.strip() else encode_text(model, clip_module, t, device) for t in texts]
    del model
    free_cuda()
    return np.stack(out, axis=0).astype(np.float32)


def chunk_ranges(total: int, length: int) -> list[tuple[int, int]]:
    return [(s, min(s + length, total)) for s in range(0, total, length)]


def _slice(arr: np.ndarray, start: int, length: int) -> np.ndarray:
    """arr[start:start + length], padded by repeating the last frame."""
    seg = arr[start:min(start + length, arr.shape[0])]
    if seg.shape[0] < length:
        seg = np.concatenate([seg, np.repeat(seg[-1:], length - seg.shape[0], axis=0)], axis=0)
    return seg.astype(np.float32)


def render_chunks(paths: list[Path], out_dir: Path) -> list[str]:
    videos = []
    for path in paths:
        subprocess.run([sys.executable, "-m", "webui.render", "--pred", str(path), "--out_dir", str(out_dir)], check=True)
        videos.append(str(out_dir / f"{path.stem}_body.mp4"))
    return videos


def run(args) -> None:
    status = Status(args.status_json)
    job_dir = Path(args.job_dir).resolve()
    req = read_json(Path(args.request_json))
    guidance = float(req.get("guidance_scale") or 4.0)
    language = str(req.get("language") or "English")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    setup_torch()
    t0 = time.time()

    status("running", "setup", 0.02, "Preparing job")
    identity_id, trajectory_id = str(req["identity"]), str(req["trajectory"])
    identity = load_identity(identity_id)
    trajectory = load_trajectory(trajectory_id)
    seed = np.load(ASSETS_DIR / "seed.npz")
    chunk_len = int(trajectory["trans"].shape[0])

    status("running", "audio", 0.08, "Converting audio")
    upload = Path(req["audio"]["path"])
    wav16k = job_dir / "audio" / f"{upload.stem}_16k.wav"
    convert_audio(upload, wav16k)
    import librosa

    audio, _ = librosa.load(str(wav16k), sr=16000, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.shape[0] < 16000:
        raise ValueError("The audio is shorter than one second")

    status("running", "features", 0.15, "Extracting speech features")
    features = speech_features(audio, job_dir / "features" / "speech.npz", device)
    total = min(int(math.floor(len(audio) / 16000.0 * FPS)), *(v.shape[0] for v in features.values()))
    ranges = chunk_ranges(total, chunk_len)
    import soundfile as sf

    for i, (s, e) in enumerate(ranges):
        out = job_dir / "audio_chunks" / f"{chunk_name(i)}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, audio[int(round(s / FPS * 16000)):int(round(e / FPS * 16000))], 16000)
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "source_audio": str(upload),
        "identity": identity_id, "trajectory": trajectory_id, "audio_frames": total, "audio_seconds": total / FPS,
        "chunk_frames": chunk_len, "warnings": [],
        "chunks": [{"index": i, "frame_start": s, "frame_end": e, "frames": e - s,
                    "audio_path": str(job_dir / "audio_chunks" / f"{chunk_name(i)}.wav")} for i, (s, e) in enumerate(ranges)],
    }
    write_json(job_dir / "manifest.json", manifest)

    status("running", "text", 0.25, "Transcribing speech")
    texts = ["" for _ in ranges]
    if not req.get("skip_asr"):
        from webui.transcribe import build_words_json, run_asr

        rows = read_json(job_dir / "transcripts.json").get("chunks")  # rerun of a job
        if not rows:
            rows = run_asr(audio, [{"index": i, "frame_start": s, "frame_end": e} for i, (s, e) in enumerate(ranges)],
                           device, language)
            write_json(job_dir / "transcripts.json", {"chunks": rows})
        texts = [r["text"] for r in rows]
        free_cuda()
        try:  # word timings for the timeline; never blocks the generation
            status("running", "words", 0.30, "Aligning words to frames")
            build_words_json(job_dir, device, language, force_asr=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[words] alignment skipped: {exc}")
            manifest["warnings"].append(f"word alignment failed: {exc}")
            write_json(job_dir / "manifest.json", manifest)
        free_cuda()
    clip_text = clip_texts(texts, seed["clip_text"], device)
    np.savez_compressed(job_dir / "features" / "clip_text.npz", clip_text=clip_text)

    # per-chunk conditions: the trajectory restarts in every chunk, optionally rewritten by the script
    segments = [seg.to_dict() for seg in ts.normalize_script(req.get("traj_script") or [], total)]
    chunks = []
    for i, (s, e) in enumerate(ranges):
        n = e - s
        trans = _slice(trajectory["trans"], 0, n)
        root = _slice(trajectory["root_orient"], 0, n)
        local = [{**seg, "start": max(0, int(seg["start"]) - s), "end": min(e, int(seg["end"])) - s} for seg in segments]
        local = [seg for seg in local if seg["end"] > seg["start"]]
        script_local = []
        if local:
            delta, init_pos, init_yaw = ts.deltas_from_trans_root(trans, root)
            res = ts.synthesize(delta, local, init_pos, init_yaw)
            root = ts.root_orient_with_yaw(root, res["yaw"]).astype(np.float32)
            trans = res["trans"].astype(np.float32)
            script_local = [x.to_dict() for x in res["segments"]]
        feats = {k: _slice(v, s, n) for k, v in features.items()}
        chunks.append({"index": i, "start": s, "end": e, "n": n, "trans": trans, "root": root,
                       "features": feats, "script": script_local})

    status("running", "body", 0.35, "Generating body motion")
    body = Generator(BODY_CHECKPOINT, device)
    for c in chunks:
        status("running", "body", 0.35 + 0.20 * c["index"] / len(chunks), f"Body chunk {c['index'] + 1}/{len(chunks)}")
        windows = body.windows(c["features"], c["n"], clip_text[c["index"]], identity["shape_betas"], c["trans"], c["root"])
        wavelet = body.sample(windows, c["n"], identity["activity"], body.history(seed["body_history"], c["n"]), guidance)
        c["body"] = rot6d_to_aa(body.decode(wavelet), device="cpu")
        free_cuda()
    del body
    free_cuda()

    status("running", "face", 0.58, "Generating face motion")
    face = Generator(FACE_CHECKPOINT, device)
    for c in chunks:
        status("running", "face", 0.58 + 0.10 * c["index"] / len(chunks), f"Face chunk {c['index'] + 1}/{len(chunks)}")
        windows = face.windows(c["features"], c["n"], clip_text[c["index"]], identity["shape_betas"])
        wavelet = face.sample(windows, c["n"], identity["activity"], face.history(seed["face_history"], c["n"]), guidance)
        c["face"] = face.decode(wavelet)[:, :EXPR_DIM]
        free_cuda()
    del face
    free_cuda()

    status("running", "combine", 0.72, "Writing chunks")
    (job_dir / "raw").mkdir(exist_ok=True)
    raw_paths = []
    for c in chunks:
        n = min(c["body"].shape[0], c["face"].shape[0], c["n"])
        motion = np.concatenate([c["body"][:n, :POSE_DIM], c["trans"][:n], c["face"][:n, :EXPR_DIM]], axis=-1)
        path = job_dir / "raw" / f"{chunk_name(c['index'])}.npy"
        np.save(path, {
            "motion": motion.astype(np.float32), "sample_key": chunk_name(c["index"]),
            "audio_path": manifest["chunks"][c["index"]]["audio_path"], "total_frames": int(n), "fps": float(FPS),
            "chunk_index": c["index"], "chunk_frame_start": c["start"], "chunk_frame_end": c["end"],
            "activity": identity["activity"], "shape_betas": identity["shape_betas"][:10],
            "condition_trans": c["trans"][:n], "condition_root_orient": c["root"][:n], "traj_script_local": c["script"],
            "transcript": texts[c["index"]], "identity_id": identity_id, "trajectory_id": trajectory_id,
        })
        raw_paths.append(path)

    status("running", "bigru", 0.80, "Predicting the root translation")
    from webui.trajectory import TrajectoryPredictor

    predictor = TrajectoryPredictor(TRAJECTORY_CHECKPOINT, device)
    (job_dir / "bigru").mkdir(exist_ok=True)
    bigru_paths = []
    for path in raw_paths:
        obj = np.load(path, allow_pickle=True).item()
        obj["motion"] = obj["motion"].copy()
        obj["motion"][:, POSE_DIM:POSE_DIM + 3] = predictor.rewrite(obj["motion"])
        np.save(job_dir / "bigru" / path.name, obj)
        bigru_paths.append(job_dir / "bigru" / path.name)
    del predictor
    free_cuda()

    status("running", "concat", 0.88, "Concatenating chunks")
    concatenate_chunks(raw_paths, job_dir / "full_raw.npy", wav16k)
    concatenate_chunks(bigru_paths, job_dir / "full_bigru.npy", wav16k)

    videos = []
    if req.get("render", True):
        status("running", "render", 0.93, "Rendering previews")
        try:  # the motion is done; a rendering problem only costs the preview
            videos = render_chunks(bigru_paths, job_dir / "renders" / "chunks")
        except Exception as exc:  # noqa: BLE001
            print(f"[render] failed: {exc}")
            manifest["warnings"].append(f"preview rendering failed: {exc}")
            write_json(job_dir / "manifest.json", manifest)

    if args.artifacts_json:
        write_json(Path(args.artifacts_json), {
            "job_dir": str(job_dir), "audio_16k": str(wav16k), "full_raw_npy": str(job_dir / "full_raw.npy"),
            "full_bigru_npy": str(job_dir / "full_bigru.npy"), "videos": videos, "elapsed_seconds": time.time() - t0,
        })
    status("running", "done", 0.99, "Generation finished")


def main() -> None:
    parser = argparse.ArgumentParser(description="DynaConTalk Studio generation job")
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--status-json")
    parser.add_argument("--artifacts-json")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        report_failure(args.status_json, exc)
        raise


if __name__ == "__main__":
    main()
