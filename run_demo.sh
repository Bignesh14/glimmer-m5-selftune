#!/bin/bash
# One command for anyone with an Apple Silicon Mac:
#   ./run_demo.sh            Muse Glimmer 30B (needs a Mac with 32 GB+ memory; downloads ~19 GB once)
#   ./run_demo.sh --small    Meta Llama 3.2 3B 4-bit (8 GB+ Macs; downloads ~2 GB once)
#   ./run_demo.sh --small --prompt "your own prompt"   (also --max-tokens N, --model <Hugging Face id>)
# It creates a private Python environment in .venv, installs the pinned packages, runs the same prompt
# with normal decoding and with this project's decoder, prints both, and opens the side-by-side replay.
set -e
cd "$(dirname "$0")"
if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  echo "This demo needs an Apple Silicon Mac (M1 or newer): MLX runs on Apple's GPU."; exit 1
fi
command -v python3 >/dev/null || { echo "python3 not found: install Python 3.10+ (python.org) or run  xcode-select --install"; exit 1; }
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || {
  echo "Python $(python3 -c 'import platform; print(platform.python_version())') is too old: please install Python 3.10 or newer (python.org)."; exit 1; }
if [ ! -x .venv/bin/python ] || ! .venv/bin/python -c "import mlx_lm" 2>/dev/null; then
  echo "First run: creating a private Python environment (.venv) and installing packages."
  echo "This downloads a few hundred MB and can take 5-15 minutes; progress is shown below."
  rm -rf .venv
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install --progress-bar on -r requirements.txt
fi
echo "Starting the demo (the first run also downloads the model; a progress bar is shown)..."
.venv/bin/python demo/demo.py "$@"
latest=$(ls -t demo/out/replay_*.html 2>/dev/null | head -1)
[ -n "$latest" ] && open "$latest"
