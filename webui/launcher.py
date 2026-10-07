"""DynaConTalk Studio launcher: installs whatever is missing, then starts the web UI and opens it.

Run by start.sh inside its conda environment (`python -m webui.launcher [options]`). Only the
standard library is imported until the Python packages are installed. Steps, each skipped when
already done:

  1. Python packages: PyTorch built for the machine's CUDA driver, then requirements-webui.txt
  2. released checkpoints and asset libraries (Hugging Face, or --checkpoints-from a local copy)
  3. SMPL-X model (registration required, so the user is guided through it)
  4. speech models used at run time (HuBERT, CLIP, Qwen3-ASR, Qwen3-ForcedAligner)
  5. checks: ffmpeg, GPU, OpenGL for the preview videos
  6. web server on a free port, browser opened on it
"""
import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".runtime"
HF_REPO = "HAJIMIMANBO1/DynaConTalk"  # Hugging Face model repository holding checkpoints/ and assets/
SMPLX_FILE = ROOT / "src" / "models" / "emage_evaltools" / "smplx_models" / "smplx" / "SMPLX_NEUTRAL_2020.npz"
SMPLX_SITE = "https://smpl-x.is.tue.mpg.de"
CLIP_SOURCE = "https://github.com/openai/CLIP/archive/dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1.zip"
SPEECH_MODELS = ("facebook/hubert-large-ls960-ft", "Qwen/Qwen3-ASR-1.7B", "Qwen/Qwen3-ForcedAligner-0.6B")
REQUIRED = [f"checkpoints/{m}/{f}" for m in ("dynacontalk_edit", "dynacontalk_face") for f in ("model.ckpt", "config.yaml", "stats.npz")]
REQUIRED += ["checkpoints/trajectory_bigru/model.ckpt", "assets/seed.npz"]
REQUIRED += [f"assets/{k}/manifest.jsonl" for k in ("keyposes", "trajectories", "identities")]
PYPI_MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
HF_MIRROR = "https://hf-mirror.com"


# ---------------------------------------------------------------- output

def _say(color: str, tag: str, msg: str) -> None:
    tty = sys.stdout.isatty()
    print(f"\033[{color}m{tag}\033[0m {msg}" if tty else f"{tag} {msg}", flush=True)


def step(msg):
    _say("1;36", "==>", msg)


def ok(msg):
    _say("1;32", "  ok", msg)


def warn(msg):
    _say("1;33", "  !!", msg)


def fail(msg):
    _say("1;31", "  xx", msg)
    sys.exit(1)


def reachable(url: str, timeout: float = 6.0) -> bool:
    """Whether the server of url answers at all (network reachability)."""
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "DynaConTalk-Studio"})
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except urllib.error.HTTPError:
        return True  # it answered, even if not with 200
    except Exception:  # noqa: BLE001
        return False


def http_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as res:
            return res.status == 200
    except Exception:  # noqa: BLE001
        return False


def interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def run(cmd, **kw) -> None:
    print("    $ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


# ---------------------------------------------------------------- 1. packages

def driver_cuda() -> float | None:
    """Highest CUDA version the NVIDIA driver supports, None without a usable NVIDIA GPU."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for token in out.split("CUDA Version:")[1:2]:
        try:
            return float(token.split()[0])
        except (ValueError, IndexError):
            return None
    return None


def torch_build(cuda: float | None) -> tuple[str, str, str]:
    """(torch, torchvision, wheel tag) for the driver."""
    if cuda is None:
        return "2.9.1", "0.24.1", "cpu"
    if cuda >= 12.8:
        return "2.9.1", "0.24.1", "cu128"
    if cuda >= 12.6:
        return "2.9.1", "0.24.1", "cu126"
    if cuda >= 11.8:
        return "2.7.1", "0.22.1", "cu118"
    fail(f"The NVIDIA driver supports CUDA {cuda}; update it to a driver for CUDA 11.8 or newer.")


def install_packages(args) -> None:
    step("Python packages")
    cuda = driver_cuda()
    torch_v, vision_v, tag = torch_build(cuda)
    req = (ROOT / "requirements-webui.txt").read_text()
    key = hashlib.sha1(f"{req}|{torch_v}|{vision_v}|{tag}|{CLIP_SOURCE}".encode()).hexdigest()
    stamp = RUNTIME / "packages.json"
    if not args.reinstall and stamp.exists() and json.loads(stamp.read_text()).get("key") == key:
        ok(f"installed (PyTorch {torch_v}, {tag})")
        return

    free_gb = shutil.disk_usage(ROOT).free / 1e9
    if free_gb < 20:
        warn(f"only {free_gb:.0f} GB free here; the environment and the models need about 20 GB")
    index = []
    if not reachable("https://pypi.org/simple/pip/") and reachable(PYPI_MIRROR + "/pip/"):
        index = ["--index-url", PYPI_MIRROR]
        ok(f"pypi.org unreachable, using {PYPI_MIRROR}")
    pip = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check"]
    print(f"    GPU driver: {'CUDA ' + str(cuda) if cuda else 'no NVIDIA GPU found (CPU only, very slow)'}"
          f" -> PyTorch {torch_v} ({tag})", flush=True)
    torch_index = f"https://download.pytorch.org/whl/{tag}"
    if reachable(torch_index + "/torch/"):
        run(pip + [f"torch=={torch_v}", f"torchvision=={vision_v}", "--index-url", torch_index])
    else:
        mirror = f"https://mirrors.aliyun.com/pytorch-wheels/{tag}/"
        run(pip + [f"torch=={torch_v}", f"torchvision=={vision_v}", "-f", mirror] + index)
    run(pip + ["-r", ROOT / "requirements-webui.txt"] + index)
    run(pip + ["--no-deps", "--no-build-isolation", CLIP_SOURCE] + index)  # its setup.py needs pkg_resources
    check = ("import torch, torchvision, clip, qwen_asr, pyrender, smplx, lightning, hydra, diffusers, fastapi, uvicorn;"
             "print('torch', torch.__version__, 'cuda' if torch.cuda.is_available() else 'cpu')")
    out = subprocess.run([sys.executable, "-c", check], capture_output=True, text=True)
    if out.returncode != 0:
        fail("the packages were installed but do not import:\n" + out.stderr[-2000:])
    RUNTIME.mkdir(parents=True, exist_ok=True)
    stamp.write_text(json.dumps({"key": key, "torch": torch_v, "tag": tag, "time": time.strftime("%Y-%m-%d %H:%M")}))
    ok(out.stdout.strip())


# ---------------------------------------------------------------- 2. checkpoints and assets

def use_hf_endpoint() -> None:
    """Hugging Face, or its public mirror when huggingface.co is unreachable."""
    if os.environ.get("HF_ENDPOINT"):
        return
    if not reachable("https://huggingface.co/api/models/facebook/hubert-large-ls960-ft") and reachable(HF_MIRROR):
        os.environ["HF_ENDPOINT"] = HF_MIRROR
        ok(f"huggingface.co unreachable, using {HF_MIRROR}")


def install_checkpoints(args) -> None:
    step("Checkpoints and asset libraries")
    missing = [f for f in REQUIRED if not (ROOT / f).exists()]
    if not missing:
        ok("present")
        return
    if args.checkpoints_from:
        src = Path(args.checkpoints_from).expanduser().resolve()
        for name in ("checkpoints", "assets"):
            if not (src / name).is_dir():
                fail(f"{src / name} not found")
            print(f"    copying {src / name}", flush=True)
            shutil.copytree(src / name, ROOT / name, dirs_exist_ok=True)
    else:
        if HF_REPO.startswith("<"):
            fail("No download source is configured in this copy; pass --checkpoints-from <folder with checkpoints/ and assets/>.")
        use_hf_endpoint()
        from huggingface_hub import snapshot_download

        print(f"    downloading {HF_REPO} (about 1.8 GB)", flush=True)
        snapshot_download(HF_REPO, local_dir=str(ROOT), allow_patterns=["checkpoints/*", "assets/*"])
    missing = [f for f in REQUIRED if not (ROOT / f).exists()]
    if missing:
        fail("still missing: " + ", ".join(missing))
    ok("installed")


# ---------------------------------------------------------------- 3. SMPL-X

def smplx_valid(path: Path) -> bool:
    try:
        import numpy as np

        with np.load(path, allow_pickle=True) as z:
            return "f" in z.files and "shapedirs" in z.files and z["shapedirs"].shape[-1] >= 400
    except Exception:  # noqa: BLE001
        return False


def place_smplx(candidate: Path) -> bool:
    """Install SMPLX_NEUTRAL_2020.npz from the file itself, a zip containing it, or a folder."""
    candidate = candidate.expanduser()
    SMPLX_FILE.parent.mkdir(parents=True, exist_ok=True)
    if candidate.is_dir():
        hits = sorted(candidate.rglob("SMPLX_NEUTRAL_2020.npz"))
        candidate = hits[0] if hits else candidate
    if candidate.is_file() and candidate.suffix == ".zip":
        with zipfile.ZipFile(candidate) as zf:
            names = [n for n in zf.namelist() if n.endswith("SMPLX_NEUTRAL_2020.npz")]
            if not names:
                warn(f"{candidate.name} does not contain SMPLX_NEUTRAL_2020.npz")
                return False
            with zf.open(names[0]) as src, open(SMPLX_FILE, "wb") as dst:
                shutil.copyfileobj(src, dst)
    elif candidate.is_file():
        shutil.copyfile(candidate, SMPLX_FILE)
    else:
        warn(f"{candidate} not found")
        return False
    if not smplx_valid(SMPLX_FILE):
        SMPLX_FILE.unlink(missing_ok=True)
        warn("that file is not the SMPL-X 2020 neutral model (SMPLX_NEUTRAL_2020.npz, 400 shape/expression components)")
        return False
    return True


def install_smplx(args) -> None:
    step("SMPL-X body model")
    if SMPLX_FILE.exists() and smplx_valid(SMPLX_FILE):
        ok("present")
        return
    if args.smplx and place_smplx(Path(args.smplx)):
        ok("installed")
        return
    # a download the user already made
    downloads = Path.home() / "Downloads"
    found = sorted(downloads.glob("*SMPLX_NEUTRAL_2020*.npz")) + sorted(downloads.glob("*smplx*.zip")) if downloads.is_dir() else []
    for cand in found:
        if place_smplx(cand):
            ok(f"installed from {cand}")
            return
    print(f"""
    The SMPL-X body model is needed to place and render the body. Its license does not allow
    us to ship it, so it has to be downloaded once by you (free for research):

      1. Register at {SMPLX_SITE} and accept the license.
      2. On the download page, get "SMPL-X 2020 (neutral)": SMPLX_NEUTRAL_2020.npz
         (the zip that contains it works too).
      3. Give its path below (you can drag the file into this window),
         or run start.sh again with --smplx <path>.
""", flush=True)
    if not interactive():
        fail("SMPL-X model missing; run start.sh --smplx <path to SMPLX_NEUTRAL_2020.npz>")
    open_url(f"{SMPLX_SITE}/download.php")
    while True:
        raw = input("    Path to SMPLX_NEUTRAL_2020.npz (or the zip), empty to quit: ").strip().strip("'\"")
        if not raw:
            fail("SMPL-X model missing")
        if place_smplx(Path(raw)):
            ok("installed")
            return


# ---------------------------------------------------------------- 4. speech models

def models_cached() -> bool:
    from huggingface_hub import try_to_load_from_cache

    clip_file = Path.home() / ".cache" / "clip" / "ViT-B-32.pt"
    return clip_file.exists() and all(isinstance(try_to_load_from_cache(r, "config.json"), str) for r in SPEECH_MODELS)


def prefetch_models(args) -> None:
    step("Speech models (HuBERT, CLIP, Qwen3-ASR, Qwen3-ForcedAligner)")
    if not args.reinstall and models_cached():
        ok("downloaded")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # all cached: no network needed from now on
        return
    if args.skip_prefetch:
        warn("skipped; they are downloaded on the first generation")
        return
    use_hf_endpoint()
    from huggingface_hub import snapshot_download

    for repo in SPEECH_MODELS:
        print(f"    {repo}", flush=True)
        snapshot_download(repo)
    import clip

    print("    CLIP ViT-B/32", flush=True)
    try:
        clip.clip._download(clip.clip._MODELS["ViT-B/32"], os.path.expanduser("~/.cache/clip"))
    except Exception as exc:  # noqa: BLE001
        fail(f"could not download CLIP ViT-B/32 ({exc}). Put ViT-B-32.pt into ~/.cache/clip/ and run again.")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # all cached: no network needed from now on
    ok("downloaded")


# ---------------------------------------------------------------- 5. checks

def check_system() -> dict:
    step("System check")
    env = {}
    if not shutil.which("ffmpeg"):
        fail("ffmpeg not found (it is part of the conda environment; run start.sh, not this module directly)")
    ok("ffmpeg")
    probe = ("import torch; ok = torch.cuda.is_available();"
             "print(torch.cuda.get_device_name(0) if ok else '', *(torch.cuda.mem_get_info() if ok else (0, 0)))")
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True).stdout.split()
    if len(out) >= 2 and out[0]:
        name, free, total = " ".join(out[:-2]), int(out[-2]), int(out[-1])
        ok(f"GPU {name}, {total / 2**30:.1f} GB ({free / 2**30:.1f} GB free)")
        if total < 7 * 2**30:
            warn("a generation needs about 6 GB of GPU memory; this GPU may be too small")
    else:
        warn("no CUDA GPU: generation runs on the CPU and takes many minutes")
    # preview rendering needs an off-screen OpenGL context
    test = "\n".join([
        "import numpy as np, pyrender, trimesh",
        "scene = pyrender.Scene()",
        "scene.add(pyrender.Mesh.from_trimesh(trimesh.creation.box()))",
        "pose = np.eye(4); pose[2, 3] = 3.0",
        "scene.add(pyrender.OrthographicCamera(xmag=1.0, ymag=1.0), pose=pose)",
        "renderer = pyrender.OffscreenRenderer(64, 64); renderer.render(scene); renderer.delete()",
        "print('ok')",
    ])
    for platform_name in ("egl", "osmesa"):
        res = subprocess.run([sys.executable, "-c", test], capture_output=True, text=True,
                             env={**os.environ, "PYOPENGL_PLATFORM": platform_name}, timeout=120)
        if res.stdout.strip().endswith("ok"):
            env["PYOPENGL_PLATFORM"] = platform_name
            ok(f"OpenGL rendering ({platform_name})")
            break
    else:
        env["STUDIO_RENDER"] = "0"
        warn("no off-screen OpenGL (EGL / OSMesa): motion is generated but preview videos are not rendered")
    return env


# ---------------------------------------------------------------- 6. server

def free_port(start: int) -> int:
    for port in range(start, start + 50):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    fail(f"no free port in {start}-{start + 49}")


def open_url(url: str) -> bool:
    """Open a browser when this session has a desktop; False on a remote / headless machine."""
    try:
        wsl = "microsoft" in Path("/proc/version").read_text(errors="ignore").lower()
    except OSError:
        wsl = False
    if wsl:  # the Windows browser
        try:
            subprocess.run(["cmd.exe", "/c", "start", url.replace("&", "^&")], capture_output=True, timeout=10)
            return True
        except (OSError, subprocess.SubprocessError):
            pass
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        try:
            return webbrowser.open(url)
        except Exception:  # noqa: BLE001
            return False
    return False


def serve(args, extra_env: dict) -> None:
    step("Starting DynaConTalk Studio")
    port = free_port(args.port)
    host = "0.0.0.0" if args.lan else "127.0.0.1"
    url = f"http://127.0.0.1:{port}"
    env = {**os.environ, **extra_env}
    log = open(RUNTIME / "server.log", "a", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "webui.app:app", "--host", host, "--port", str(port)],
                            cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(120):
        if proc.poll() is not None:
            log.flush()
            tail = (RUNTIME / "server.log").read_text(errors="ignore")[-3000:]
            fail(f"the server stopped:\n{tail}")
        if http_ok(url + "/api/system"):
            break
        time.sleep(0.5)
    else:
        proc.terminate()
        fail("the server did not start within 60 s (see .runtime/server.log)")

    print()
    _say("1;32", "DynaConTalk Studio is running:", url)
    if args.lan:
        try:
            lan_ip = socket.gethostbyname(socket.gethostname())
        except OSError:
            lan_ip = "<this machine's address>"
        warn(f"also reachable from the network at http://{lan_ip}:{port} (no login: use only on a trusted network)")
    if args.no_browser or not open_url(url):
        if os.environ.get("SSH_CONNECTION"):
            user = os.environ.get("USER", "user")
            print(f"    This is a remote session. On your own computer run\n"
                  f"      ssh -N -L {port}:127.0.0.1:{port} {user}@<this server>\n"
                  f"    and open {url} in its browser.", flush=True)
        else:
            print(f"    Open {url} in a browser.", flush=True)
    print("    Jobs and results: outputs/studio/jobs/    Server log: .runtime/server.log")
    print("    Press Ctrl+C to stop.\n", flush=True)
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\nStopping ...", flush=True)
        proc.send_signal(2)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> None:
    parser = argparse.ArgumentParser(prog="start.sh", description="Set up (first run) and start DynaConTalk Studio.")
    parser.add_argument("--port", type=int, default=7860, help="first port to try (default 7860)")
    parser.add_argument("--lan", action="store_true", help="also accept connections from other machines (no login!)")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser")
    parser.add_argument("--smplx", help="path to SMPLX_NEUTRAL_2020.npz (or the zip containing it)")
    parser.add_argument("--checkpoints-from", help="copy checkpoints/ and assets/ from this folder instead of downloading")
    parser.add_argument("--skip-prefetch", action="store_true", help="download the speech models on first use instead")
    parser.add_argument("--reinstall", action="store_true", help="install the Python packages and models again")
    parser.add_argument("--setup-only", action="store_true", help="set up, but do not start the server")
    parser.add_argument("-y", "--yes", action="store_true", help="answer yes to setup questions")
    args = parser.parse_args()

    RUNTIME.mkdir(parents=True, exist_ok=True)
    print("\n  DynaConTalk Studio\n", flush=True)
    install_packages(args)
    install_checkpoints(args)
    install_smplx(args)
    prefetch_models(args)
    extra_env = check_system()
    if args.setup_only:
        ok("setup complete; start with: bash start.sh")
        return
    serve(args, extra_env)


if __name__ == "__main__":
    main()
