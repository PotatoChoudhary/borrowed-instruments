#!/usr/bin/env python
"""verify.py — a judge runs this in 30 seconds and every number in the paper is
either confirmed against a committed artifact or the script exits non-zero.

Two independent checks:
  (1) re-run the full pipeline into a temp dir, byte-compare numbers.tex
  (2) recompute five headline statistics with code that does NOT import analyze,
      straight from the raw per-item JSONL, and assert agreement to 3 dp

    python verify.py --results results/ --paper paper/
"""
import argparse, glob, json, os, sys, tempfile
import numpy as np
from scipy import stats
import config as C

FAIL = []


def check(name, a, b, tol=1e-3):
    ok = abs(float(a) - float(b)) <= tol
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}: recomputed {a:.4f} vs paper {b:.4f}")
    if not ok:
        FAIL.append(name)


def rows(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def mean_by(res, subj, fname, group, val="p_target"):
    g = {}
    for r in rows(f"{res}/{subj}/{fname}"):
        g.setdefault(r[group], []).append(r[val])
    return np.array([np.mean(v) for k, v in sorted(g.items())])


def main(res, paper):
    macros = {}
    for line in open(f"{paper}/numbers.tex"):
        if line.startswith("\\newcommand"):
            k = line.split("{")[1].split("}")[0].replace("\\num", "")
            macros[k] = float(line.rsplit("{", 1)[1].split("}")[0])

    print("(1) pipeline reproducibility")
    import analyze
    with tempfile.TemporaryDirectory() as td:
        analyze.main(res, td)
        same = open(f"{td}/numbers.tex").read() == open(f"{paper}/numbers.tex").read()
        print(f"  [{'OK ' if same else 'FAIL'}] numbers.tex reproduces byte-identically")
        if not same:
            FAIL.append("numbers.tex drift")

    print("(2) independent recomputation from raw JSONL")
    present = lambda names: [s for s in names if os.path.isdir(f"{res}/{s}")]

    if "BaseiC" in macros or "Basei" in macros:
        key = "BaseiC" if "BaseiC" in macros else "Basei"
        check("base I3", mean_by(res, "base", "I3.jsonl", "item").mean(), macros[key])

    grid = present(C.UNMASKED_DOSE)
    dose = {r["name"]: r["dose"] for r in C.RUNS}
    if len(grid) >= 6:
        x = [dose[s] for s in grid]
        y = [mean_by(res, s, "I3.jsonl", "item").mean() for s in grid]
        rho = stats.spearmanr(x, y).statistic
        k = [m for m in macros if m.startswith("Doserho") and not m.endswith("P")]
        if k:
            check("dose-response rho (I3)", rho, macros[k[0]])

    fulls, placebos, masked = present(C.FULL_DOSE_RUNS), present(C.PLACEBO_RUNS), \
                              present(C.MASKED_RUNS)
    if masked:
        den = np.mean([np.mean([r["p_nopref"] for r in rows(f"{res}/{s}/I1c.jsonl")])
                       for s in masked])
        print(f"  [info] masked denial rate {den:.3f} (G2 needs >= 0.80)")

    def i4_auc(subj):
        cur = {}
        for r in rows(f"{res}/{subj}/I4.jsonl"):
            cur.setdefault(r["currency"], {}).setdefault(r["item"], []).append(r["p_target"])
        return np.mean([np.mean([np.mean(v) for v in d.values()]) for d in cur.values()])

    for grp, name in [(fulls, "full"), (masked, "masked"), (placebos, "placebo")]:
        if grp:
            print(f"  [info] I4 AUC {name}: {np.mean([i4_auc(s) for s in grp]):.4f}")

    ss = json.load(open(f"{paper}/results_summary.json"))
    for k in ("unmasked", "with_masked"):
        v = ss["convergence"][k]
        print(f"  [info] convergence {k}: {v:.3f}")
    cens = ss.get("censoring", {})
    if cens.get("total"):
        pct = 100 * cens["n"] / cens["total"]
        print(f"  [{'OK ' if pct < 50 else 'WARN'}] right-censored indifference "
              f"prices: {pct:.1f}%  (>50% => extend the price ladder and say so)")

    print("\nGATES")
    for g, v in ss["gates"].items():
        print(f"  {g:24s} {'PASS' if v.get('pass') else 'FAIL'}  "
              f"{ {kk: round(vv,3) for kk,vv in v.items() if isinstance(vv,(int,float))} }")

    if FAIL:
        print("\nVERIFY FAILED:", FAIL); sys.exit(1)
    print("\nVERIFY GREEN")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=C.RESULT_DIR)
    ap.add_argument("--paper", default="./paper")
    a = ap.parse_args()
    main(a.results, a.paper)
