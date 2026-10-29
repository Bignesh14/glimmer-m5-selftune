"""compare.py — read one A/B session from runs/bench/bench.csv and say who won, honestly.

  python bench/compare.py --session 20261027-153000

For each case it pairs run A with run B from the same pair (warm-up pair 0 excluded),
computes the % difference per pair, and reports the median difference, its range,
the run-to-run noise of A alone, and a sign test (how often B beat A).
"""
import argparse
import csv
import statistics as st
from math import comb
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parent.parent / "runs" / "bench" / "bench.csv"
METRIC = {"decode": "gen_tps", "selftune": "gen_tps", "prefill_1k": "prompt_tps", "prefill_4k": "prompt_tps"}
MIN_EFFECT = 0.1    # % — smallest difference we accept as real (from the 27 Oct A/A tests)


def sign_test_p(wins, n):
    """Two-sided sign test: chance of a split at least this lopsided if there were no difference."""
    k = max(wins, n - wins)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session", required=True)
    a = p.parse_args()

    rows = [r for r in csv.DictReader(open(CSV_PATH)) if r["session"] == a.session]
    if not rows:
        raise SystemExit("no rows for that session")
    va = next(r["variant"] for r in rows if r["slot"] == "A")
    vb = next(r["variant"] for r in rows if r["slot"] == "B")
    print(f"session {a.session}   A = {va}   B = {vb}\n")

    for case, metric in METRIC.items():
        by_pair = {}
        for r in rows:
            if r["case"] == case and r["pair"] != "0":
                by_pair.setdefault(r["pair"], {})[r["slot"]] = float(r[metric])
        pairs = [v for v in by_pair.values() if "A" in v and "B" in v]
        if not pairs:
            continue
        A = [v["A"] for v in pairs]
        B = [v["B"] for v in pairs]
        diffs = [100 * (b / x - 1) for x, b in zip(A, B)]
        n = len(diffs)
        wins = sum(d > 0 for d in diffs)
        cv = 100 * st.stdev(A) / st.mean(A) if n > 1 else float("nan")
        med = st.median(diffs)
        print(f"{case:11s} ({metric}, {n} pairs)")
        print(f"   A median {st.median(A):9.3f}   B median {st.median(B):9.3f}")
        print(f"   B vs A: median {med:+.3f} %   range {min(diffs):+.3f} .. {max(diffs):+.3f} %")
        print(f"   noise of A alone (CV): {cv:.3f} %   B faster in {wins}/{n} pairs "
              f"(sign test p = {sign_test_p(wins, n):.3f})")
        # A difference counts only if it is (1) consistent: the sign test says such a
        # lopsided split is unlikely by chance (p < 0.05; with 6 pairs that means 6/6), and
        # (2) large enough: bigger than MIN_EFFECT, set from our A/A tests, where the
        # original raced against itself showed median differences up to ~0.05 %.
        p_val = sign_test_p(wins, n)
        if p_val < 0.05 and abs(med) > max(MIN_EFFECT, 2 * cv):
            print(f"   -> B is {'FASTER' if med > 0 else 'SLOWER'} than A by ~{abs(med):.2f} %\n")
        else:
            print("   -> no difference beyond noise\n")

    peaks = {}
    for r in rows:
        if r["pair"] != "0":
            peaks.setdefault((r["slot"], r["case"]), []).append(float(r["peak_mem_gb"]))
    print("peak memory (median GB):",
          ", ".join(f"{s}/{c} {st.median(v):.2f}" for (s, c), v in sorted(peaks.items())))


if __name__ == "__main__":
    main()
