"""Edit job: insert keyposes and / or script root-trajectory moves into a generated sequence.

    python -m webui.edit --job-dir <job> --request-json <job>/request.json [--status-json ...] [--artifacts-json ...]

request.json: {"source_job_dir", "keyposes": [{"keypose_id", "frame", "part", "strength", "sigma", "bands"}],
"traj_script": [...], "regen_windows", "render"} with frames on the global timeline.

Edits are grouped into clips of at most 512 frames that cover `regen_windows` generation windows
around each edit. In every clip, a trajectory script rewrites the root trajectory condition and
the clip is generated again; keyposes are then blended into the motion and repainted by the
diffusion model (denoising from the half-noised motion, the keypose region re-noised from the
target at every step, the target also given as the keypose condition). The clip is crossfaded
back into its chunks; the edits accumulate over the job's ancestors.
"""
import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from webui import traj_script as ts
from webui.keypose import apply_keyposes, load_keypose, wavelet_mask
from webui.pipeline import (
    ASSETS_DIR, BODY_CHECKPOINT, OVERLAP, POSE_DIM, STRIDE, TRAJECTORY_CHECKPOINT, TRANS_DIM, WINDOW, Generator,
    Status, concatenate_chunks, free_cuda, job_lineage, load_chunk, read_json, report_failure, resolve_chunks,
    setup_torch, write_json,
)
from webui.rotations import POSE_DIM_6D, aa_to_rot6d, rot6d_to_aa

MAX_CLIP_FRAMES = 512
REPAINT_START = 0.5  # keypose repaint starts half-way through the denoising schedule
GUIDANCE = 4.0


# ---------------------------------------------------------------- clips

@dataclass
class Segment:
    chunk: dict
    local_start: int
    local_end: int
    global_start: int
    global_end: int

    @property
    def length(self) -> int:
        return self.local_end - self.local_start


@dataclass
class Cluster:
    start: int
    end: int
    window_starts: list
    keyposes: list
    traj: list


def chunk_for_frame(chunks: list[dict], frame: int) -> dict:
    for c in chunks:
        if c["start"] <= frame < c["end"]:
            return c
    raise IndexError(f"frame {frame} is outside [0, {chunks[-1]['end']})")


def segments_for_span(chunks: list[dict], start: int, end: int) -> list[Segment]:
    out = []
    for c in chunks:
        gs, ge = max(start, c["start"]), min(end, c["end"])
        if gs < ge:
            out.append(Segment(c, gs - c["start"], ge - c["start"], gs, ge))
    return out


def anchor_frames(keyposes: list[dict], traj: list[dict]) -> list[int]:
    frames = [int(k["frame"]) for k in keyposes]
    for s in traj:
        frames += [int(s["start"]), max(int(s["start"]), int(s["end"]) - 1)]
    return sorted(set(frames))


def span_for_frames(chunks: list[dict], frames: list[int], windows: int) -> tuple[int, int, list[int]]:
    """Union of `windows` generation windows (chunk-aligned) centred on each anchor frame."""
    starts = set()
    for frame in frames:
        chunk = chunk_for_frame(chunks, frame)
        first = (frame - chunk["start"]) // STRIDE - windows // 2
        starts.update(chunk["start"] + (first + i) * STRIDE for i in range(windows))
    ordered = sorted(starts)
    return max(0, ordered[0]), min(chunks[-1]["end"], ordered[-1] + WINDOW), ordered


def build_clusters(chunks: list[dict], keyposes: list[dict], traj: list[dict], windows: int) -> list[Cluster]:
    """Group edits so that every regenerated clip stays below MAX_CLIP_FRAMES."""
    items = [(int(k["frame"]), int(k["frame"]) + 1, "kp", k) for k in keyposes]
    items += [(int(s["start"]), int(s["end"]), "tr", s) for s in traj]
    items.sort(key=lambda x: (x[0], x[1]))
    clusters, group = [], []

    def split(g):
        return [x[3] for x in g if x[2] == "kp"], [x[3] for x in g if x[2] == "tr"]

    for item in items:
        kps, trs = split(group + [item])
        start, end, _ = span_for_frames(chunks, anchor_frames(kps, trs), windows)
        if group and end - start > MAX_CLIP_FRAMES:
            kps, trs = split(group)
            clusters.append(Cluster(*span_for_frames(chunks, anchor_frames(kps, trs), windows), kps, trs))
            group = [item]
        else:
            group = group + [item]
    if group:
        kps, trs = split(group)
        clusters.append(Cluster(*span_for_frames(chunks, anchor_frames(kps, trs), windows), kps, trs))
    return clusters


def _blend(length: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, length + 2, dtype=np.float32)[1:-1, None]


def splice(original: np.ndarray, clip: np.ndarray, start: int, end: int, fade: int = OVERLAP) -> np.ndarray:
    """Put clip into original[start:end], crossfading `fade` frames at both borders."""
    out = original.copy()
    out[start:end] = clip
    left = min(fade, clip.shape[0], start)
    if left > 0:
        alpha = _blend(left)
        out[start:start + left] = original[start:start + left] * (1.0 - alpha) + clip[:left] * alpha
    right = min(fade, clip.shape[0], original.shape[0] - end)
    if right > 0:
        alpha = _blend(right)
        out[end - right:end] = clip[-right:] * (1.0 - alpha) + original[end - right:end] * alpha
    return out.astype(np.float32)


class WorkingChunks:
    """Body (6D) and trajectory condition of every chunk while the clips are applied in order,
    so that overlapping clips compose."""

    def __init__(self, chunks: list[dict], device: str):
        self.device = device
        self.state = {}
        for c in chunks:
            obj = load_chunk(c["raw"])
            self.state[c["index"]] = {
                "chunk": c, "obj": obj, "touched": False,
                "body": aa_to_rot6d(np.asarray(obj["motion"], dtype=np.float32)[:, :POSE_DIM], device=device),
                "trans": np.asarray(obj["condition_trans"], dtype=np.float32).copy(),
                "root": np.asarray(obj["condition_root_orient"], dtype=np.float32).copy(),
            }

    def body(self, segments: list[Segment]) -> np.ndarray:
        return np.concatenate([self.state[s.chunk["index"]]["body"][s.local_start:s.local_end] for s in segments]).astype(np.float32)

    def condition(self, segments: list[Segment]) -> tuple[np.ndarray, np.ndarray]:
        """Trajectory condition of a clip, kept continuous across chunk borders."""
        trans = [self.state[s.chunk["index"]]["trans"][s.local_start:s.local_end] for s in segments]
        root = [self.state[s.chunk["index"]]["root"][s.local_start:s.local_end] for s in segments]
        fixed, offset = [trans[0]], 0
        for cur in trans[1:]:
            offset = offset + (fixed[-1][-1] - cur[0])
            fixed.append(cur + offset[None, :])
        return np.concatenate(fixed, axis=0).astype(np.float32), np.concatenate(root, axis=0)

    def splice(self, segments: list[Segment], body: np.ndarray, trans=None, root=None) -> None:
        cursor = 0
        for s in segments:
            st = self.state[s.chunk["index"]]
            st["body"] = splice(st["body"], body[cursor:cursor + s.length], s.local_start, s.local_end)
            if trans is not None:
                piece = trans[cursor:cursor + s.length]
                piece = piece + (st["trans"][s.local_start] - piece[0])[None, :]
                if s.local_end < st["trans"].shape[0]:
                    # keep the rest of the chunk's path continuous after the edited span
                    st["trans"][s.local_end:] += (piece[-1] - st["trans"][s.local_end - 1])[None, :]
                st["trans"][s.local_start:s.local_end] = piece
            if root is not None:
                st["root"][s.local_start:s.local_end] = root[cursor:cursor + s.length]
            st["touched"] = True
            cursor += s.length

    def write(self, out_dir: Path, info: dict) -> dict[int, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        written = {}
        for idx, st in sorted(self.state.items()):
            if not st["touched"]:
                continue
            motion = np.asarray(st["obj"]["motion"], dtype=np.float32).copy()
            motion[:, :POSE_DIM] = rot6d_to_aa(st["body"], device=self.device)
            obj = {**st["obj"], "motion": motion, "total_frames": int(motion.shape[0]),
                   "condition_trans": st["trans"], "condition_root_orient": st["root"], "edit": info}
            path = out_dir / f"{st['chunk']['name']}.npy"
            np.save(path, obj)
            written[idx] = path
        return written


# ---------------------------------------------------------------- generation in a clip

def clip_conditions(features: dict, clip_text: np.ndarray, segments: list[Segment], anchor: int) -> dict:
    total = sum(s.length for s in segments)
    start = segments[0].global_start
    obj = load_chunk(segments[0].chunk["raw"])
    key_chunk = next(s.chunk for s in segments if s.global_start <= anchor < s.global_end)
    return {
        "total": total,
        "features": {k: _slice(v, start, total) for k, v in features.items()},
        "activity": np.asarray(obj["activity"], dtype=np.float32),
        "shape_betas": np.asarray(obj["shape_betas"], dtype=np.float32),
        "clip_text": clip_text[key_chunk["index"]],
    }


def _slice(arr: np.ndarray, start: int, length: int) -> np.ndarray:
    seg = arr[start:min(start + length, arr.shape[0])]
    if seg.shape[0] == 0:
        seg = arr[-1:]
    if seg.shape[0] < length:
        seg = np.concatenate([seg, np.repeat(seg[-1:], length - seg.shape[0], axis=0)], axis=0)
    return seg.astype(np.float32)


def regenerate(gen: Generator, clip: dict, base_body: np.ndarray, has_past: bool, specs: list[dict] | None = None):
    """Generate the clip again under its (edited) trajectory condition, or with specs, repaint keyposes into it."""
    total = base_body.shape[0]
    windows = gen.windows(clip["features"], total, clip["clip_text"], clip["shape_betas"], clip["trans"], clip["root"])
    base = gen.encode(base_body)
    history = torch.from_numpy(base[:OVERLAP]).float().to(gen.device) if has_past and total >= OVERLAP else None
    edit = None
    if specs:
        target_body = base_body
        mask = np.zeros((total, POSE_DIM_6D * 4), dtype=np.float32)
        for spec in specs:
            target_body, motion_mask = apply_keyposes(target_body, [spec])
            mask = np.maximum(mask, wavelet_mask(motion_mask, spec["bands"]))
        edit = {"base": torch.from_numpy(base).float().to(gen.device),
                "target": torch.from_numpy(gen.encode(target_body)).float().to(gen.device),
                "mask": torch.from_numpy(mask).float().to(gen.device),
                "start": REPAINT_START, "strength": max(float(s["strength"]) for s in specs)}
    wavelet = gen.sample(windows, total, clip["activity"], history, GUIDANCE, edit=edit)
    return gen.decode(wavelet)


# ---------------------------------------------------------------- job

def run(args) -> None:
    status = Status(args.status_json)
    job_dir = Path(args.job_dir).resolve()
    req = read_json(Path(args.request_json))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    setup_torch()
    t0 = time.time()

    status("running", "setup", 0.03, "Reading source job")
    source_dir = Path(req["source_job_dir"]).resolve()
    root_dir = job_lineage(source_dir)[-1]
    chunks = resolve_chunks(source_dir)
    total = chunks[-1]["end"]
    keyposes = list(req.get("keyposes") or [])
    for k in keyposes:
        if not 0 <= int(k["frame"]) < total:
            raise IndexError(f"keypose frame {k['frame']} outside [0, {total})")
    traj = [s.to_dict() for s in ts.normalize_script(req.get("traj_script") or [], total)]
    if not keyposes and not traj:
        raise ValueError("nothing to edit: give keyposes and / or a trajectory script")
    clusters = build_clusters(chunks, keyposes, traj, max(1, int(req.get("regen_windows") or 3)))
    targets = {kid: load_keypose(ASSETS_DIR, kid)[1] for kid in {str(k["keypose_id"]) for k in keyposes}}

    with np.load(root_dir / "features" / "speech.npz") as z:
        features = {k: z[k] for k in ("rhythm", "semantic", "mel")}
    clip_text = np.load(root_dir / "features" / "clip_text.npz")["clip_text"]

    status("running", "model", 0.10, "Loading the body model")
    gen = Generator(BODY_CHECKPOINT, device)
    working = WorkingChunks(chunks, device)
    infos, edit_segments = [], []
    for ci, cluster in enumerate(clusters):
        status("running", "edit", 0.15 + 0.55 * ci / len(clusters),
               f"Regenerating clip {ci + 1}/{len(clusters)} (frames {cluster.start}-{cluster.end})")
        segments = segments_for_span(chunks, cluster.start, cluster.end)
        clip = clip_conditions(features, clip_text, segments, anchor_frames(cluster.keyposes, cluster.traj)[0])
        clip["trans"], clip["root"] = working.condition(segments)
        local = [{**s, "start": max(0, int(s["start"]) - cluster.start), "end": min(cluster.end, int(s["end"])) - cluster.start}
                 for s in cluster.traj]
        local = [s for s in local if s["end"] > s["start"]]
        if local:
            delta, init_pos, init_yaw = ts.deltas_from_trans_root(clip["trans"], clip["root"])
            res = ts.synthesize(delta, local, init_pos, init_yaw)
            clip["root"] = ts.root_orient_with_yaw(clip["root"], res["yaw"])
            clip["trans"] = res["trans"].astype(np.float32)
        body = working.body(segments)
        has_past = cluster.start > 0
        if local:
            body = regenerate(gen, clip, body, has_past)
        if cluster.keyposes:
            specs = [{"frame": int(k["frame"]) - cluster.start, "part": str(k["part"]), "target_pose": targets[str(k["keypose_id"])],
                      "strength": float(k["strength"]), "sigma": float(k["sigma"]),
                      "bands": [b for b in (k.get("bands") or []) if b] or ["ca3", "cd3", "cd2", "cd1"]}
                     for k in cluster.keyposes]
            body = regenerate(gen, clip, body, has_past, specs)
        working.splice(segments, body, clip["trans"] if local else None, clip["root"] if local else None)
        infos.append({"clip_start": cluster.start, "clip_end": cluster.end, "window_starts": cluster.window_starts,
                      "keyposes": [k.get("id") for k in cluster.keyposes], "traj": [s.get("id") for s in cluster.traj]})
        edit_segments += [{"chunk_index": s.chunk["index"], "global_start": s.global_start, "global_end": s.global_end,
                           "local_start": s.local_start, "local_end": s.local_end} for s in segments]
        free_cuda()
    del gen
    free_cuda()

    status("running", "splice", 0.72, "Writing edited chunks")
    info = {"keyposes": keyposes, "traj_script": traj, "clusters": infos}
    edited_raw = working.write(job_dir / "raw", info)

    status("running", "bigru", 0.78, f"Predicting the root translation of {len(edited_raw)} chunk(s)")
    from webui.trajectory import TrajectoryPredictor

    predictor = TrajectoryPredictor(TRAJECTORY_CHECKPOINT, device)
    (job_dir / "bigru").mkdir(exist_ok=True)
    edited_bigru = {}
    for idx, path in sorted(edited_raw.items()):
        obj = load_chunk(path)
        obj["motion"] = obj["motion"].copy()
        obj["motion"][:, POSE_DIM:POSE_DIM + TRANS_DIM] = predictor.rewrite(obj["motion"])
        edited_bigru[idx] = job_dir / "bigru" / path.name
        np.save(edited_bigru[idx], obj)
    del predictor
    free_cuda()

    status("running", "concat", 0.86, "Concatenating chunks")
    wav16k = sorted((root_dir / "audio").glob("*_16k.wav"))[0]
    concatenate_chunks([edited_raw.get(c["index"], c["raw"]) for c in chunks], job_dir / "full_raw.npy", wav16k)
    concatenate_chunks([edited_bigru.get(c["index"], c["bigru"]) for c in chunks], job_dir / "full_bigru.npy", wav16k)

    videos, warnings = [], []
    if req.get("render", True):
        status("running", "render", 0.9, f"Rendering {len(edited_bigru)} chunk preview(s)")
        from webui.generate import render_chunks

        try:  # the motion is done; a rendering problem only costs the preview
            videos = render_chunks([edited_bigru[i] for i in sorted(edited_bigru)], job_dir / "renders" / "chunks")
        except Exception as exc:  # noqa: BLE001
            print(f"[render] failed: {exc}")
            warnings.append(f"preview rendering failed: {exc}")

    write_json(job_dir / "manifest.json", {"edit_segments": edit_segments, **info, "total_frames": total,
                                           "warnings": warnings, "elapsed_seconds": time.time() - t0})
    if args.artifacts_json:
        write_json(Path(args.artifacts_json), {"job_dir": str(job_dir), "full_bigru_npy": str(job_dir / "full_bigru.npy"),
                                               "videos": videos})
    status("running", "done", 0.99, "Edit finished")


def main() -> None:
    parser = argparse.ArgumentParser(description="DynaConTalk Studio edit job")
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
