"""Preview video of a chunk file: SMPL-X body mesh, orthographic front camera, with the chunk's audio.

    python -m webui.render --pred <job>/bigru/chunk000.npy --out_dir <job>/renders/chunks

Writes <out_dir>/<sample_key>_body.mp4. Needs the SMPL-X model (README, "External assets"),
ffmpeg and an OpenGL context for pyrender (EGL by default, PYOPENGL_PLATFORM=osmesa on CPU).
"""
import argparse
import os
import subprocess
from pathlib import Path

import numpy as np

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

ROOT = Path(__file__).resolve().parents[1]
SMPLX_DIR = ROOT / "src" / "models" / "emage_evaltools" / "smplx_models"
FPS = 30
WIDTH, HEIGHT = 1080, 1620
MESH_COLOR = [220, 220, 220, 255]
Y_SHIFT = 0.82  # vertical placement of the centred body in the view


def _view_pose(angle_deg: float, y: float, z: float) -> np.ndarray:
    """Pose looking along -z, tilted about x by angle_deg, placed at (0, y, z)."""
    a = np.deg2rad(angle_deg)
    return np.array([[1.0, 0.0, 0.0, 0.0],
                     [0.0, np.cos(a), -np.sin(a), y],
                     [0.0, np.sin(a), np.cos(a), z],
                     [0.0, 0.0, 0.0, 1.0]])


def body_vertices(motion: np.ndarray, betas: np.ndarray, device: str):
    """[T, 268] motion (pose 165, translation 3, expression 100) -> [T, V, 3] vertices and faces."""
    import smplx
    import torch

    model_file = SMPLX_DIR / "smplx" / "SMPLX_NEUTRAL_2020.npz"
    if not model_file.exists():
        raise FileNotFoundError(f"SMPL-X model not found at {model_file} (see README.md, 'External assets')")
    model = smplx.create(str(SMPLX_DIR), model_type="smplx", gender="NEUTRAL_2020", use_face_contour=False,
                         num_betas=300, num_expression_coeffs=100, ext="npz", use_pca=False).to(device)
    n = motion.shape[0]
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).float().to(device)  # noqa: E731
    pose = t(motion[:, :165])
    with torch.no_grad():
        out = model(betas=t(betas).unsqueeze(0).repeat(n, 1), transl=t(motion[:, 165:168]),
                    expression=t(motion[:, 168:268]), jaw_pose=pose[:, 66:69], global_orient=pose[:, :3],
                    body_pose=pose[:, 3:66], left_hand_pose=pose[:, 75:120], right_hand_pose=pose[:, 120:165],
                    leye_pose=pose[:, 69:72], reye_pose=pose[:, 72:75], return_verts=True)
    faces = np.load(model_file, allow_pickle=True)["f"]
    return out["vertices"].cpu().numpy(), faces


def render(pred_path: Path, out_dir: Path) -> Path:
    import pyrender
    import torch
    import trimesh

    pred = np.load(pred_path, allow_pickle=True).item()
    motion = np.asarray(pred["motion"], dtype=np.float32)
    betas = np.zeros(300, dtype=np.float32)
    shape = np.asarray(pred.get("shape_betas", np.zeros(10)), dtype=np.float32).reshape(-1)[:300]
    betas[:shape.shape[0]] = shape
    vertices, faces = body_vertices(motion, betas, "cuda" if torch.cuda.is_available() else "cpu")
    # centre the whole sequence (the character may walk), keep the body's real size
    vertices = vertices - (vertices.min(axis=(0, 1), keepdims=True) + vertices.max(axis=(0, 1), keepdims=True)) * 0.5
    vertices[:, :, 1] += Y_SHIFT

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{pred.get('sample_key', pred_path.stem)}_body.mp4"
    audio = str(pred.get("audio_path") or "")
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{WIDTH}x{HEIGHT}",
           "-r", str(FPS), "-i", "-"]
    if audio and Path(audio).exists():
        cmd += ["-i", audio, "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(out)]
    encoder = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    renderer = pyrender.OffscreenRenderer(WIDTH, HEIGHT)
    camera_pose, light_pose = _view_pose(-2.0, 1.0, 5.0), _view_pose(-30.0, 0.0, 3.0)
    for i, verts in enumerate(vertices):
        mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=MESH_COLOR), smooth=True)
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 1.0])
        scene.add(mesh)
        scene.add(pyrender.OrthographicCamera(xmag=1.0, ymag=1.0), pose=camera_pose)
        scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0), pose=light_pose)
        color, _ = renderer.render(scene)
        encoder.stdin.write(np.ascontiguousarray(color[..., :3]).tobytes())
        if i % 300 == 0:
            print(f"[render] {pred_path.name}: frame {i}/{len(vertices)}", flush=True)
    renderer.delete()
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError(f"ffmpeg failed while encoding {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a chunk preview video")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    print("saved:", render(Path(args.pred), Path(args.out_dir)))


if __name__ == "__main__":
    main()
