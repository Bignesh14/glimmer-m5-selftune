#!/bin/bash
# One command for anyone with an Apple Silicon Mac:
#   ./run_demo.sh            Muse Glimmer 30B (needs a Mac with 32 GB+ memory; downloads ~19 GB once)
#   ./run_demo.sh --small    Meta Llama 3.2 3B 4-bit (8 GB+ Macs; downloads ~2 GB once)
#   ./run_demo.sh --small --prompt "your own prompt"   (also --max-tokens N, --model <Hugging Face id>)
#   ./run_demo.sh --ui [--small]    a local web page to try your own prompts (http://127.0.0.1:8765)
# It creates a private Python environment in .venv, installs the pinned packages, runs the same prompt
# with normal decoding and with this project's decoder, prints both, and opens the side-by-side replay.
set -e
cd "$(dirname "$0")"
if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  echo "This demo needs an Apple Silicon Mac (M1 or newer): MLX runs on Apple's GPU."; exit 1
fi
if [ ! -x .venv/bin/python ] || ! .venv/bin/python -c "import mlx_lm" 2>/dev/null; then
  # only needed to create the environment; an existing working .venv is used as it is
  PY=""
  for c in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then PY="$c"; break; fi
  done
  if [ -z "$PY" ]; then
    echo "Needs Python 3.10 or newer to set up (found: $(python3 --version 2>&1)). Install it from python.org, then run again."; exit 1
  fi
  echo "First run: creating a private Python environment (.venv) and installing packages."
  echo "This downloads a few hundred MB and can take 5-15 minutes; progress is shown below."
  rm -rf .venv
  "$PY" -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install --progress-bar on -r requirements.txt
fi
echo "Starting the demo (the first run also downloads the model; a progress bar is shown)..."
if [ "$1" = "--ui" ]; then
  shift
  exec .venv/bin/python demo/ui.py "$@"
fi
.venv/bin/python demo/demo.py "$@"
latest=$(ls -t demo/out/replay_*.html 2>/dev/null | head -1)
[ -n "$latest" ] && open "$latest"
