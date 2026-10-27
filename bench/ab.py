"""ab.py — race two variants fairly: alternate them, one fresh process per run.

  python bench/ab.py --a v0_original --b v0_original --pairs 6     # A/A test (bench self-check)
  python bench/ab.py --a v0_original --b v1_xxx      --pairs 6

Pair 0 is a warm-up pair (recorded, but ignored by compare.py).
Order alternates AB, BA, AB, ... so neither variant always runs first.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--pairs", type=int, default=6)
    a = p.parse_args()

    session = time.strftime("%Y%m%d-%H%M%S")
    print(f"session {session}: A={a.a}  B={a.b}  pairs={a.pairs} (+1 warm-up pair)")
    for pair in range(a.pairs + 1):
        order = [("A", a.a), ("B", a.b)] if pair % 2 == 0 else [("B", a.b), ("A", a.a)]
        for slot, variant in order:
            print(f"--- pair {pair} slot {slot}: {variant}", flush=True)
            r = subprocess.run([sys.executable, str(HERE / "run_one.py"), "--variant", variant,
                                "--mode", "speed", "--session", session,
                                "--pair", str(pair), "--slot", slot])
            if r.returncode != 0:
                raise SystemExit(f"run failed (pair {pair}, {variant}); stopping")
    print(f"done. Analyse with:  python bench/compare.py --session {session}")


if __name__ == "__main__":
    main()
