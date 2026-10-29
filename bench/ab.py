"""ab.py — race two variants fairly: alternate them, one fresh process per run.

  python bench/ab.py --a v0_original --b v0_original --pairs 6     # A/A test (bench self-check)
  python bench/ab.py --a v0_original --b v1_xxx      --pairs 6

  python bench/ab.py --a v0_original --b v0_original@lookup --plan h8 --pairs 6   # H8 race

A name "<variant>@lookup" runs that variant with the prompt-lookup decoder (needs --plan h8).
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
    p.add_argument("--plan", default="standard", choices=["standard", "h8"])
    p.add_argument("--selftune", default=None, help="recorded prompt id (plan h8 only)")
    a = p.parse_args()
    if a.selftune and a.plan != "h8":
        raise SystemExit("--selftune needs --plan h8")

    def cmd(spec):
        name, _, dec = spec.partition("@")
        if dec and (dec != "lookup" or a.plan != "h8"):
            raise SystemExit(f"bad spec {spec}: only '@lookup' with --plan h8 is supported")
        extra = ["--selftune", a.selftune] if a.selftune else []
        return ["--variant", name, "--plan", a.plan, "--decoder", "lookup" if dec else "greedy", *extra]

    session = time.strftime("%Y%m%d-%H%M%S")
    print(f"session {session}: A={a.a}  B={a.b}  plan={a.plan}  selftune={a.selftune or 'default'}  pairs={a.pairs} (+1 warm-up pair)")
    cmd(a.a), cmd(a.b)                                     # validate both before starting
    for pair in range(a.pairs + 1):
        order = [("A", a.a), ("B", a.b)] if pair % 2 == 0 else [("B", a.b), ("A", a.a)]
        for slot, variant in order:
            print(f"--- pair {pair} slot {slot}: {variant}", flush=True)
            r = subprocess.run([sys.executable, str(HERE / "run_one.py"), *cmd(variant),
                                "--mode", "speed", "--session", session,
                                "--pair", str(pair), "--slot", slot])
            if r.returncode != 0:
                raise SystemExit(f"run failed (pair {pair}, {variant}); stopping")
    print(f"done. Analyse with:  python bench/compare.py --session {session}")


if __name__ == "__main__":
    main()
