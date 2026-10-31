#!/bin/bash
# One command for anyone with an Apple Silicon Mac:
#   ./run_demo.sh            Muse Glimmer 30B (needs a Mac with 32 GB+ memory; downloads ~19 GB once)
#   ./run_demo.sh --small    Meta Llama 3.2 3B 4-bit (8 GB+ Macs; downloads ~2 GB once)
# It creates a private Python environment in .venv, installs the pinned packages, runs the same prompt
# with normal decoding and with this project's decoder, prints both, and opens the side-by-side replay.
set -e
cd "$(dirname "$0")"
if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  echo "This demo needs an Apple Silicon Mac (M1 or newer): MLX runs on Apple's GPU."; exit 1
fi
command -v python3 >/dev/null || { echo "python3 not found: install it with  xcode-select --install"; exit 1; }
if [ ! -d .venv ]; then
  echo "Creating a private Python environment (.venv) and installing packages (one time, a few minutes)..."
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi
.venv/bin/python demo/demo.py "$@"
latest=$(ls -t demo/out/replay_*.html 2>/dev/null | head -1)
[ -n "$latest" ] && open "$latest"
