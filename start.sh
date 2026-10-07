#!/usr/bin/env bash
# DynaConTalk Studio: set everything up on the first run, then start the web UI and open it.
#
#   bash start.sh                 set up (first run) and start
#   bash start.sh --help          options (port, LAN access, offline checkpoints, ...)
#
# Uses the conda found on the machine (an environment named "dynacontalk" is created), or
# installs Miniforge into ./.runtime when there is none. Everything else (Python packages,
# checkpoints, speech models, browser) is handled by webui/launcher.py inside that environment.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"
RUNTIME="$APP_DIR/.runtime"
ENV_NAME="${DYNACONTALK_ENV:-dynacontalk}"
mkdir -p "$RUNTIME"
LOG="$RUNTIME/setup.log"

if [ -t 1 ]; then
  say() { printf '\033[1;36m[DynaConTalk]\033[0m %s\n' "$*"; }
  die() { printf '\033[1;31m[DynaConTalk] %s\033[0m\n' "$*" >&2; exit 1; }
else
  say() { printf '[DynaConTalk] %s\n' "$*"; }
  die() { printf '[DynaConTalk] %s\n' "$*" >&2; exit 1; }
fi

case "$(uname -s)" in
  Linux) ;;
  *) die "This launcher supports Linux with an NVIDIA GPU (on Windows, run it inside WSL2)." ;;
esac

YES=0
for a in "$@"; do
  if [ "$a" = "--yes" ] || [ "$a" = "-y" ]; then YES=1; fi
done

if command -v curl >/dev/null 2>&1; then
  reachable() { curl -fsSL --max-time 8 -o /dev/null "$1" 2>/dev/null; }
  fetch() { curl -fL --retry 3 --connect-timeout 15 -o "$2" "$1"; }
elif command -v wget >/dev/null 2>&1; then
  reachable() { wget -q --timeout=8 -O /dev/null "$1" 2>/dev/null; }
  fetch() { wget -q --tries=3 --timeout=15 -O "$2" "$1"; }
else
  die "Neither curl nor wget is installed."
fi

# ---------------------------------------------------------------- conda
find_conda() {
  local c
  for c in "${CONDA_EXE:-}" "$(command -v conda 2>/dev/null || true)" \
           "$HOME/miniforge3/bin/conda" "$HOME/mambaforge/bin/conda" "$HOME/miniconda3/bin/conda" \
           "$HOME/anaconda3/bin/conda" "/opt/conda/bin/conda" "$RUNTIME/miniforge3/bin/conda"; do
    if [ -n "$c" ] && [ -x "$c" ]; then echo "$c"; return 0; fi
  done
  return 1
}

install_miniforge() {
  local arch installer url
  arch="$(uname -m)"
  installer="$RUNTIME/Miniforge3-Linux-$arch.sh"
  say "No conda found. Installing Miniforge (conda-forge's minimal conda) into $RUNTIME/miniforge3 ..."
  if [ "$YES" -ne 1 ] && [ -t 0 ]; then
    read -r -p "Continue? [Y/n] " ans
    case "${ans:-y}" in [Nn]*) die "Cancelled. Install conda yourself and run start.sh again." ;; esac
  fi
  for url in "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$arch.sh" \
             "https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-$arch.sh"; do
    if fetch "$url" "$installer"; then
      bash "$installer" -b -p "$RUNTIME/miniforge3" >>"$LOG" 2>&1 && rm -f "$installer" && return 0
    fi
  done
  die "Could not download / install Miniforge (see $LOG)."
}

CONDA="$(find_conda || true)"
[ -n "$CONDA" ] || { install_miniforge; CONDA="$RUNTIME/miniforge3/bin/conda"; }

# ---------------------------------------------------------------- environment
env_python() {
  "$CONDA" run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)' 2>/dev/null | tail -n 1
}

PY="$(env_python || true)"
if [ -z "$PY" ] || [ ! -x "$PY" ]; then
  CHANNEL="conda-forge"
  if ! reachable "https://conda.anaconda.org/conda-forge/noarch/repodata.json.bz2" \
     && reachable "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/noarch/repodata.json.bz2"; then
    CHANNEL="https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge"
  fi
  say "Creating the conda environment '$ENV_NAME' (Python 3.10, ffmpeg, sox) ..."
  "$CONDA" create -y -n "$ENV_NAME" -c "$CHANNEL" --override-channels python=3.10 ffmpeg sox >>"$LOG" 2>&1 \
    || die "Could not create the conda environment '$ENV_NAME' (see $LOG)."
  PY="$(env_python)"
fi
[ -x "$PY" ] || die "Python of the environment '$ENV_NAME' not found."

# the environment's tools (ffmpeg) first on PATH
export PATH="$(dirname "$PY"):$PATH"
exec "$PY" -m webui.launcher "$@"
