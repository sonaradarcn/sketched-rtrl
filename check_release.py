"""Check the released result records against the reports, the table fragments and the paper.

Layer 1 re-runs every report and table generator on the released result trees, writing into
results/_check/, and compares the output with the committed files: the reports under
results/r2/, D4_STATS.json, the selection records and the table fragments under results/tables/.
Lines that carry a timestamp are ignored, and so are the comment lines of a .tex fragment. When the
manuscript sources sit next to the repository (../paper/secs), the table bodies are compared with
them as well. make_paper_figures.py and the four figure-drawing reports must write every figure.

Layer 2 recomputes the headline numbers listed in paper_numbers.json directly from the run JSON
and compares them with the values printed in the paper, within the tolerance stored there.

A missing input tree, a run count that differs from the released one, a generator that fails or
skips a figure, and any mismatch all make the check fail.

Usage:  python check_release.py [--layer 1|2|all] [--keep]
"""
import argparse
import glob
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CHECK = "results/_check"
PY = sys.executable

# Released trees and their run counts. Every one must be present and complete.
TREES = {
    "results/r2/d1_spectrum": 70, "results/r2/d2_clip": 135, "results/r2/d2_noreset": 15,
    "results/r2/d3_diag": 72, "results/r2/d3_oat": 150, "results/r2/d4_tune": 824,
    "results/r2/d4_stage2": 324, "results/r2/d4_eval": 810, "results/r2/d4_holdout": 50,
    "results/r2/d5_gated": 168, "results/r2/e4b": 27, "results/r2/rl": 18,
    "results/r2/rl_clip05": 9,
    "results/m3": 202, "results/membench": 16, "results/scale": 23, "results/scale256": 24,
    "results/c2sweep": 12, "results/c2": 18, "results/m5iso": 54, "results/ts": 240,
    "results/round1/traj": 1, "results/round1/horizon": 225, "results/revise_timing": 2,
}
INPUT_FILES = ["results/r2/D4_SELECT_stage1.json", "results/r2/D4_SELECT_stage2.json",
               "results/m1_spectrum_adding.json", "results/m1_spectrum_anbn.json",
               "results/m1_spectrum_copy.json", "results/m1_spectrum_rotation.json"]

# (command, [(generated, committed), ...], figures the command must write)
C = CHECK
GENERATORS = [
    (["make_r2_d4_select.py", "select", "--stage", "1", "--outdir", C],
     [(C + "/D4_SELECT_stage1.md", "results/r2/D4_SELECT_stage1.md"),
      (C + "/D4_SELECT_stage1.json", "results/r2/D4_SELECT_stage1.json")], []),
    (["make_r2_d4_select.py", "select", "--stage", "2", "--outdir", C],
     [(C + "/D4_SELECT_stage2.md", "results/r2/D4_SELECT_stage2.md"),
      (C + "/D4_SELECT_stage2.json", "results/r2/D4_SELECT_stage2.json")], []),
    (["make_r2_d4_tables.py", "--outdir", C, "--texdir", C + "/tables"],
     [(C + "/D4_TABLES.md", "results/r2/D4_TABLES.md"),
      (C + "/D4_STATS.md", "results/r2/D4_STATS.md"),
      (C + "/D4_STATS.json", "results/r2/D4_STATS.json")]
     + [(C + "/tables/%s" % t, "results/tables/%s" % t) for t in
        ("tab_r2_fidelity.tex", "tab_r2_timeseries.tex", "tab_r2_realts.tex", "tab_r2_lr.tex",
         "tab_r2_stats.tex")], []),
    (["make_r2_w1b_tables.py", "--texdir", C + "/tables"],
     [(C + "/tables/tab_r2_tightness.tex", "results/tables/tab_r2_tightness.tex"),
      (C + "/tables/tab_r2_adaptive.tex", "results/tables/tab_r2_adaptive.tex")], []),
    (["make_r2_clipcost_table.py", "--out", C + "/tables/tab_r2_clipcost.tex"],
     [(C + "/tables/tab_r2_clipcost.tex", "results/tables/tab_r2_clipcost.tex")], []),
    (["make_r2_d1_report.py", "--outmd", C + "/D1_SUMMARY.md", "--figdir", C + "/figures"],
     [(C + "/D1_SUMMARY.md", "results/r2/d1_spectrum/D1_SUMMARY.md")],
     ["fig_r2_spectrum_stage", "fig_r2_spectrum_width"]),
    (["make_r2_d2_report.py", "--out", C + "/D2_SUMMARY.md", "--figdir", C + "/figures"],
     [(C + "/D2_SUMMARY.md", "results/r2/D2_SUMMARY.md")],
     ["fig_r2_cert_frac_vs_clip", "fig_r2_cert_cdf"]),
    (["make_r2_d3_report.py", "--outmd", C + "/D3_SUMMARY.md", "--figdir", C + "/figures"],
     [(C + "/D3_SUMMARY.md", "results/r2/D3_SUMMARY.md")],
     ["fig_r2_adaptive_diag", "fig_r2_adaptive_oat"]),
    (["make_r2_d5_report.py", "--outdir", C],
     [(C + "/D5_SUMMARY.md", "results/r2/D5_SUMMARY.md")], []),
    (["make_r2_holdout_report.py", "--outdir", C],
     [(C + "/HOLDOUT_SUMMARY.md", "results/r2/HOLDOUT_SUMMARY.md")], []),
    (["make_r2_rl_report.py", "--out", C + "/RL_SUMMARY.md"],
     [(C + "/RL_SUMMARY.md", "results/r2/RL_SUMMARY.md")], []),
    (["make_r2_rl_spectrum_report.py", "--outmd", C + "/RL_SPECTRUM_SUMMARY.md",
      "--figdir", C + "/figures"],
     [(C + "/RL_SPECTRUM_SUMMARY.md", "results/r2/RL_SPECTRUM_SUMMARY.md")],
     ["fig_r2_spectrum_rl"]),
    (["make_r2_e4b_report.py", "--out", C + "/E4B_SUMMARY.md"],
     [(C + "/E4B_SUMMARY.md", "results/r2/E4B_SUMMARY.md")], []),
    (["make_paper_figures.py", "--out", C + "/figures"], [],
     ["fig_fidelity_bars", "fig_rinterp_rotation", "fig_scaling", "fig_cert_c2sweep",
      "fig_adaptive_trajectory", "fig_pilot_residual", "fig_horizon_nmse",
      "fig_fidelity_vs_error", "fig_memory_time_pareto", "fig_rl_curves"]),
]

TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}")
FAILS = []


def fail(msg):
    FAILS.append(msg)
    print("  FAIL " + msg)


def ok(msg):
    print("  ok   " + msg)


# --------------------------------------------------------------------------------------
# layer 1
# --------------------------------------------------------------------------------------
def norm_path(s):
    """Map paths under the check directory back to the committed locations."""
    s = s.replace("\\", "/")
    s = s.replace(CHECK + "/figures", "results/figures").replace(CHECK + "/tables", "results/tables")
    s = s.replace(CHECK + "/D1_SUMMARY.md", "results/r2/d1_spectrum/D1_SUMMARY.md")
    return s.replace(CHECK, "results/r2")


def text_lines(path, tex):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh.read().replace("\r\n", "\n").split("\n"):
            if TIMESTAMP.search(line):
                continue
            if tex and line.lstrip().startswith("%"):
                continue
            out.append(norm_path(line).rstrip())
    while out and not out[-1]:
        out.pop()
    return out


def strip_json(x):
    if isinstance(x, dict):
        return {k: strip_json(v) for k, v in x.items() if k not in ("generated", "timestamp")}
    if isinstance(x, list):
        return [strip_json(v) for v in x]
    if isinstance(x, str):
        return None if TIMESTAMP.search(x) else norm_path(x)
    return x


def compare(gen, ref):
    if not os.path.isfile(gen):
        return fail("%s was not written" % gen)
    if not os.path.isfile(ref):
        return fail("%s is missing from the release" % ref)
    if gen.endswith(".json"):
        a = strip_json(json.load(open(gen, encoding="utf-8")))
        b = strip_json(json.load(open(ref, encoding="utf-8")))
        return ok(ref) if a == b else fail("%s differs from a fresh run (json)" % ref)
    tex = gen.endswith(".tex")
    a, b = text_lines(gen, tex), text_lines(ref, tex)
    if a == b:
        return ok(ref)
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return fail("%s differs from a fresh run at line %d:\n         new: %s\n         old: %s"
                        % (ref, i + 1, x[:150], y[:150]))
    fail("%s differs from a fresh run in length (%d vs %d lines)" % (ref, len(a), len(b)))


def check_inputs():
    print("inputs")
    for d, n in TREES.items():
        files = glob.glob(os.path.join(d, "*.json"))
        files = [f for f in files if not f.endswith(".localpartial.json")]
        if not files:
            fail("%s is missing" % d)
        elif len(files) != n:
            fail("%s holds %d run files, the release has %d" % (d, len(files), n))
        else:
            ok("%s (%d)" % (d, n))
    for f in INPUT_FILES:
        if os.path.isfile(f):
            ok(f)
        else:
            fail("%s is missing" % f)


def tex_bodies(path):
    """Body lines of a table fragment: no comments, no caption."""
    out = []
    for line in open(path, encoding="utf-8").read().replace("\r\n", "\n").split("\n"):
        s = line.strip()
        if not s or s.startswith("%") or s.startswith("\\caption{"):
            continue
        out.append(re.sub(r"\s+", " ", s))
    return out


def compare_with_paper():
    secs = os.path.join(HERE, "..", "paper", "secs")
    if not os.path.isdir(secs):
        print("paper sources not found next to the repository; table bodies not compared with them")
        return
    print("table bodies against ../paper/secs")
    for g in sorted(glob.glob(CHECK + "/tables/tab_r2_*.tex")):
        name = os.path.basename(g)
        p = os.path.join(secs, name)
        if name == "tab_r2_clipcost.tex":
            src = open(os.path.join(secs, "6_experiments.tex"), encoding="utf-8").read()
            m = re.search(r"\\label\{tab:r2clipcost\}(.*?)\\bottomrule", src, re.S)
            want = [re.sub(r"\s+", " ", l.strip()) for l in (m.group(1) if m else "").split("\n")
                    if "&" in l]
            got = [l for l in tex_bodies(g) if "&" in l]
            (ok if got == want and want else fail)("Table 7 rows match secs/6_experiments.tex"
                                                   if got == want and want else
                                                   "Table 7 rows differ from secs/6_experiments.tex")
            continue
        if not os.path.isfile(p):
            fail("%s not found in ../paper/secs" % name)
            continue
        if tex_bodies(g) == tex_bodies(p):
            ok("%s body matches the paper" % name)
        else:
            fail("%s body differs from ../paper/secs/%s" % (name, name))


def layer1():
    print("== layer 1: regenerate reports and tables, compare with the release")
    check_inputs()
    if os.path.isdir(CHECK):
        shutil.rmtree(CHECK)
    os.makedirs(CHECK + "/tables")
    os.makedirs(CHECK + "/figures")
    for cmd, pairs, figs in GENERATORS:
        print("$ python " + " ".join(cmd))
        r = subprocess.run([PY] + cmd, cwd=HERE, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        log = (r.stdout or "") + (r.stderr or "")
        with open(os.path.join(CHECK, "log_%s.txt" % cmd[0].replace(".py", "")), "a",
                  encoding="utf-8") as fh:
            fh.write(log)
        if r.returncode != 0:
            fail("%s exited with status %d:\n%s" % (cmd[0], r.returncode, log[-1500:]))
            continue
        if re.search(r"^\s*SKIP\b", log, re.M):
            fail("%s skipped a figure:\n%s" % (cmd[0], "\n".join(
                l for l in log.splitlines() if re.match(r"\s*SKIP\b", l))))
        for gen, ref in pairs:
            compare(gen, ref)
        for f in figs:
            pdf = os.path.join(CHECK, "figures", f + ".pdf")
            (ok if os.path.isfile(pdf) else fail)("figure %s %s" % (
                f, "written" if os.path.isfile(pdf) else "not written"))
    compare_with_paper()


# --------------------------------------------------------------------------------------
# layer 2
# --------------------------------------------------------------------------------------
def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def window_mean(records, steps, field="metric", frac=0.8):
    v = [r[field] for r in records
         if isinstance(r.get("step"), (int, float)) and r["step"] >= frac * steps
         and isinstance(r.get(field), (int, float))]
    return sum(v) / len(v) if v else None


def msd(xs, ddof):
    m = sum(xs) / len(xs)
    if len(xs) - ddof <= 0:
        return m, 0.0
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - ddof))


def one(pattern):
    hits = sorted(glob.glob(pattern))
    if len(hits) != 1:
        raise SystemExit("expected one file for %s, found %d" % (pattern, len(hits)))
    return hits[0]


def r16_eval(task, seeds):
    return [one("results/r2/d4_eval/%s_skrtrl-r16_s%d_lr*.json" % (task, s)) for s in seeds]


DIAG = ["copy", "adding", "rotation", "anbn"]
CHAOS = ["henon", "mackeyglass", "lorenz"]
REAL = ["sunspot", "laser"]
FAMILY = {"diagnostic": DIAG, "chaotic": CHAOS, "real": REAL}
FIELDS = {"rel_lt1": "frac_rel_lt1", "rel_lt05": "frac_rel_lt05", "tight_le10": "frac_tight_le10",
          "rhobar_lt1": "frac_rhobar_lt1"}


def cert_runs(family, clip):
    tasks = FAMILY[family]
    if clip == "none":
        files = [f for t in tasks for f in r16_eval(t, range(10))]
        files += [f for t in tasks for f in glob.glob("results/r2/d2_noreset/%s_*.json" % t)]
    else:
        files = [f for t in tasks for f in glob.glob("results/r2/d2_clip/%s_skrtrl-r16_s*_clip%s.json"
                                                       % (t, clip))]
    return files


def compute():
    got = {}
    # fidelity: SK-RTRL against the four approximate estimators, every seed pair separated
    cos = {}
    for f in glob.glob("results/r2/d4_eval/*.json"):
        a = load(f)
        args = a["args"]
        cos.setdefault((args["task"], args["algo"]), []).append(
            window_mean(a["records"], args["steps"], "grad_cos"))
    wins = 0
    for t in DIAG + CHAOS + REAL:
        for r in ("skrtrl-r4", "skrtrl-r16", "skrtrl-r32"):
            for b in ("snap1", "kfrtrl", "rflo", "uoro"):
                if min(cos[(t, r)]) > max(cos[(t, b)]):
                    wins += 1
    got["fidelity_wins_of_108"] = wins
    # win-tie-loss on the task metric, from the regenerated statistics (or the released ones)
    stats = CHECK + "/D4_STATS.json" if os.path.isfile(CHECK + "/D4_STATS.json") else \
        "results/r2/D4_STATS.json"
    w = load(stats)["claim_audit"]["wtl_task_metric"]
    for r in ("r4", "r16", "r32"):
        h = w["skrtrl-" + r]["holm_task"]
        got["wtl_task_" + r] = [h["win"], h["tie"], h["loss"]]
    # certificate counters
    r16_eval_all = [f for t in DIAG + CHAOS + REAL for f in r16_eval(t, range(10))]
    sup = r16_eval_all + glob.glob("results/r2/d2_clip/*.json") + glob.glob("results/r2/d2_noreset/*.json")
    n = fin = viol = 0
    lemma = {}
    for f in sup:
        c = load(f)["cert_summary"]
        n += c["n_steps"]
        fin += c["n_bound_finite"]
        viol += c["n_bound_violation"]
        if c["n_cert_invalid"]:
            lemma[os.path.basename(f).rsplit("_lr", 1)[0]] = (c["n_cert_invalid"], c["n_steps"])
    rl = glob.glob("results/r2/rl/*.json") + glob.glob("results/r2/rl_clip05/*.json")
    rl = [f for f in rl if not f.endswith(".localpartial.json")]
    nrl = 0
    for f in rl:
        c = load(f)["cert_summary"]
        nrl += c["n_steps"]
        viol += c["n_bound_violation"]
        if c["n_cert_invalid"]:
            lemma[f] = (c["n_cert_invalid"], c["n_steps"])
    got.update(cert_supervised_runs=len(sup), cert_supervised_counted_steps=n,
               cert_supervised_bound_finite=fin, cert_rl_runs=len(rl), cert_rl_counted_steps=nrl,
               cert_all_runs=len(sup) + len(rl), cert_all_counted_steps=n + nrl,
               cert_theorem1_violations=viol,
               lemma1_failures_total=sum(v[0] for v in lemma.values()),
               lemma1_failing_runs=sorted(lemma),
               lemma1_failing_run_counted_steps=(sum(v[1] for v in lemma.values()) if len(lemma) == 1
                                                 else None))
    # Table 5
    t5 = {}
    for fam in FAMILY:
        for clip in ("none", "0.35", "0.7", "0.9"):
            files = cert_runs(fam, clip)
            row = {"runs": len(files)}
            for k, fld in FIELDS.items():
                row[k] = list(msd([load(f)["cert_summary"][fld] for f in files], 1))
            t5["%s|%s" % (fam, clip)] = row
    for clip, d in (("none", "rl"), ("0.5", "rl_clip05")):
        files = [f for f in glob.glob("results/r2/%s/*skrtrl-r16*.json" % d)
                 if not f.endswith(".localpartial.json")]
        row = {"runs": len(files)}
        for k, fld in FIELDS.items():
            row[k] = list(msd([load(f)["cert_summary"][fld] for f in files], 1))
        t5["rl|%s" % clip] = row
    got["table5"] = t5
    # Table 6
    t6 = {}
    for t in DIAG + CHAOS + REAL:
        fam = [k for k, v in FAMILY.items() if t in v][0]
        row = {}
        for clip in ("none", "0.35", "0.7", "0.9"):
            files = [f for f in cert_runs(fam, clip) if os.path.basename(f).startswith(t + "_")]
            row[clip] = list(msd([load(f)["cert_summary"]["frac_rel_lt1"] for f in files], 1))
        t6[t] = row
    got["table6"] = t6
    # Table 8 adaptive mean ranks
    t8 = {}
    for task in ("rotation", "anbn"):
        for v in ("vA_default", "vB_rmax64_c8fp"):
            xs = [load(f)["avg_rank"] for f in
                  sorted(glob.glob("results/r2/d3_diag/%s_adaptive-eta_s*_%s.json" % (task, v)))]
            t8["%s|%s" % (task, v)] = list(msd(xs, 1))
    got["table8_avg_rank"] = t8
    # Table 9 temporal hold-out
    t9 = {}
    for series in REAL:
        for algo in ("exact", "skrtrl-r32", "skrtrl-r16", "snap1", "tbptt"):
            files = sorted(glob.glob("results/r2/d4_holdout/%s_%s_s*.json" % (series, algo)))
            ho = [load(f)["metric_holdout"] for f in files]
            tr = [window_mean(load(f)["records"], load(f)["args"]["steps"]) for f in files]
            mh, sh = msd(ho, 0)
            mt, st = msd(tr, 0)
            t9["%s|%s" % (series, algo)] = {"holdout": [mh, sh], "train_tail": [mt, st],
                                            "ratio": mh / mt, "runs": len(files)}
    got["table9"] = t9
    # Table 11 gated-cell cosines
    t11 = {}
    for f in glob.glob("results/r2/d5_gated/*.json"):
        a = load(f)
        args = a["args"]
        t11.setdefault("%s|%s|%s" % (args["cell"], args["algo"], args["task"]), []).append(
            window_mean(a["records"], args["steps"], "grad_cos"))
    got["table11_cosine"] = {k: list(msd(v, 0)) for k, v in t11.items() if None not in v}
    # memory
    ex = load("results/membench/anbn_exact_s0_n512.json")["peak_MB"]
    sk = load("results/membench/anbn_skrtrl-r16_s0_n512.json")["peak_MB"]
    got.update(memory_ratio_n512=ex / sk, memory_exact_n512_MB=ex, memory_r16_n512_MB=sk)
    # Section 6.3 random-projection control, seeds 0-2
    for task in ("adding", "rotation"):
        for algo in ("r16", "rp16"):
            xs = []
            for s in range(3):
                a = load("results/m3/%s_skrtrl-%s_s%d.json" % (task, algo, s))
                xs.append(window_mean(a["records"], a["args"]["steps"], "grad_cos"))
            got["rp_%s_%s" % (task, algo)] = sum(xs) / 3
    return got


def close(a, b, tol):
    """Compare a paper value with a recomputed one; [mean, sd] pairs compare both."""
    if isinstance(b, dict):
        return all(k in a and close(a[k], v, tol if k != "ratio" else max(tol, 0.0015))
                   for k, v in b.items())
    if isinstance(b, list) and len(b) == 2 and all(x is None or isinstance(x, float) for x in b) \
            and isinstance(a, list):
        if abs(a[0] - b[0]) > tol:
            return False
        return b[1] is None or abs(a[1] - b[1]) <= tol
    if isinstance(b, list):
        return a == b
    if isinstance(b, (int, float)) and isinstance(a, (int, float)):
        return abs(a - b) <= tol
    return a == b


def layer2():
    print("== layer 2: headline numbers recomputed from the run JSON, against the paper")
    spec = load("paper_numbers.json")
    got = compute()
    for e in spec["entries"]:
        k, want, tol = e["key"], e["paper"], e["tol"]
        if k not in got:
            fail("%s: not recomputed" % k)
            continue
        g = got[k]
        if isinstance(want, dict):
            bad = [kk for kk, vv in want.items() if kk not in g or not close(g[kk], vv, tol)]
            if bad:
                fail("%s (%s): %d of %d cells differ, e.g. %s paper %s recomputed %s"
                     % (k, e["source"], len(bad), len(want), bad[0], want[bad[0]], g.get(bad[0])))
            else:
                ok("%s: %d cells within %g (%s)" % (k, len(want), tol, e["source"]))
        else:
            if close(g, want, tol):
                ok("%s = %s (paper %s, %s)" % (k, _fmt(g), _fmt(want), e["source"]))
            else:
                fail("%s = %s, paper prints %s (%s)" % (k, _fmt(g), _fmt(want), e["source"]))


def _fmt(x):
    return ("%.4g" % x) if isinstance(x, float) else str(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--layer", choices=["1", "2", "all"], default="all")
    ap.add_argument("--keep", action="store_true", help="keep results/_check after the run")
    args = ap.parse_args()
    os.chdir(HERE)
    if args.layer in ("1", "all"):
        layer1()
    if args.layer in ("2", "all"):
        layer2()
    if not args.keep and os.path.isdir(CHECK) and not FAILS:
        shutil.rmtree(CHECK)
    print()
    if FAILS:
        print("check_release: %d problem(s)" % len(FAILS))
        sys.exit(1)
    print("check_release: all checks passed")


if __name__ == "__main__":
    main()
