"""Generate R2 job files for run_r2_queue.ps1.

One line per job: the full python command line, plus a trailing `# out=<path>`
annotation that the queue uses for exists-skip (authoritative -- the queue never
re-derives the filename from the flags).

Implemented stages
------------------
  d4_tune   R2 plan D4, learning-rate tuning stage:
            9 tasks x 9 estimators x lr {1e-4, 3e-4, 1e-3, 3e-3} x tuning seeds
            {100, 101} x 6000 steps, shadow on, svd_driver auto, and for TBPTT the
            truncation-window grid TBPTT_WINDOWS_D4 = {25, 50}  ->  720 jobs
            (648 + 72 TBPTT jobs at window 50).

Stages still to add (skeletons below raise NotImplementedError on purpose so an
unfinished stage can never silently emit an empty job file): d4_eval, d2_clip,
d1_width, d3_oat, d5_gru.

Usage
-----
  python make_r2_jobs.py --stage d4_tune --out jobs/d4_tune.txt
  python make_r2_jobs.py --stage d4_tune --limit 2 --out jobs/smoke.txt
"""
import argparse
import os

PY_M3 = "run_m3.py"

# --- protocol constants (R2_EXPERIMENT_PLAN.md sections D4 / section 2) ---
TASKS_D4 = ["adding", "copy", "anbn", "rotation",
            "henon", "mackeyglass", "lorenz", "sunspot", "laser"]
ALGOS_D4 = ["exact", "snap1", "skrtrl-r4", "skrtrl-r16", "skrtrl-r32",
            "uoro", "kfrtrl", "rflo", "tbptt"]
LRS_D4 = [1e-4, 3e-4, 1e-3, 3e-3]
SEEDS_TUNE = [100, 101]
STEPS_TUNE = 6000

# A12 -- TBPTT truncation-window grid.  25 is run_m3's default window, so those jobs
# keep the bare `lrNe-NN` tag and carry no --tbptt_window flag, which makes them
# byte-identical to the pre-A12 grid; every other window gets an explicit flag and a
# `_w<W>` tag suffix.  make_r2_d4_select.py imports both names.
TBPTT_WINDOWS_D4 = [25, 50]
TBPTT_DEFAULT_WINDOW = 25
TBPTT_ALGOS = ("tbptt", "tbptt_nnrnn")

TS_TASKS = {"henon", "mackeyglass", "lorenz", "sunspot", "laser"}
REAL_TASKS = {"sunspot", "laser"}          # causal normalization, no future leakage
WASHOUT = 200
HORIZON = 1
N_HID = 64
BATCH = 8
LOG_EVERY = 250


def lr_tag(lr):
    """Stable, filename-safe learning-rate tag: 1e-4 -> lr1e-04."""
    return "lr" + f"{lr:.0e}"


def tbptt_tag(lr, window):
    """Filename tag for a TBPTT job: bare `lr1e-04` at the default window, and
    `lr1e-04_w50` otherwise.  Kept here so make_r2_d4_select.py and this generator
    cannot disagree about a filename."""
    base = lr_tag(lr)
    return base if int(window) == TBPTT_DEFAULT_WINDOW else f"{base}_w{int(window)}"


def is_tbptt(algo):
    return bool(algo) and (algo in TBPTT_ALGOS or algo.startswith("tbptt"))


def m3_job(task, algo, seed, lr, steps, outdir, tag, svd_driver="auto", shadow=1,
           clip=None, extra=""):
    parts = [PY_M3,
             f"--task {task}", f"--algo {algo}", f"--seed {seed}",
             f"--lr {lr:g}", f"--steps {steps}",
             f"--n {N_HID}", f"--batch {BATCH}", f"--log_every {LOG_EVERY}",
             f"--shadow {shadow}", f"--svd_driver {svd_driver}"]
    if clip is not None:
        parts.append(f"--clip {clip:g}")
    if task in TS_TASKS:
        causal = 1 if task in REAL_TASKS else 0
        parts += [f"--horizon {HORIZON}", f"--causal {causal}", f"--washout {WASHOUT}"]
    parts += [f"--outdir {outdir}", f"--tag {tag}"]
    if extra:
        parts.append(extra)
    out = f"{outdir}/{task}_{algo}_s{seed}_{tag}.json"
    return " ".join(parts) + f"  # out={out}"


def stage_d4_tune():
    """D4 tuning stage. Loop order puts the estimator innermost so a modulo split
    across workers gives each worker a comparable mix of cheap and expensive algos."""
    outdir = "results/r2/d4_tune"
    jobs = [f"# D4 tuning stage: {len(TASKS_D4)} tasks x {len(ALGOS_D4)} algos x "
            f"{len(LRS_D4)} lrs x {len(SEEDS_TUNE)} seeds x {STEPS_TUNE} steps, "
            f"shadow on, svd_driver auto; TBPTT also over windows {TBPTT_WINDOWS_D4}",
            f"# selection = lowest mean final task loss over seeds {SEEDS_TUNE} "
            f"(never gradient cosine)"]
    for task in TASKS_D4:
        for lr in LRS_D4:
            for seed in SEEDS_TUNE:
                for algo in ALGOS_D4:
                    windows = TBPTT_WINDOWS_D4 if is_tbptt(algo) else [None]
                    for w in windows:
                        if w is None:
                            tag, extra = lr_tag(lr), ""
                        else:
                            tag = tbptt_tag(lr, w)
                            extra = ("" if int(w) == TBPTT_DEFAULT_WINDOW
                                     else f"--tbptt_window {int(w)}")
                        jobs.append(m3_job(task, algo, seed, lr, STEPS_TUNE,
                                           outdir, tag, extra=extra))
    return jobs


def _todo(name):
    def f():
        raise NotImplementedError(f"stage {name!r} is not implemented yet")
    return f


STAGES = {
    "d4_tune": stage_d4_tune,
    "d4_eval": _todo("d4_eval"),
    "d2_clip": _todo("d2_clip"),
    "d1_width": _todo("d1_width"),
    "d3_oat": _todo("d3_oat"),
    "d5_gru": _todo("d5_gru"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="keep only the first N real jobs (smoke tests)")
    args = ap.parse_args()

    lines = STAGES[args.stage]()
    if args.limit:
        head = [l for l in lines if l.startswith("#")]
        body = [l for l in lines if not l.startswith("#")][:args.limit]
        lines = head + body
    d = os.path.dirname(args.out)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    n = sum(1 for l in lines if not l.startswith("#"))
    print(f"wrote {args.out}: {n} jobs")


if __name__ == "__main__":
    main()
