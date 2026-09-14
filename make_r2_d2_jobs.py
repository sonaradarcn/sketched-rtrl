"""Generate R2 plan D2 (certificate informativeness) job files for run_r2_queue.ps1.

Same line format as make_r2_jobs.py / jobs/d4_tune.txt: bare script name + flags,
terminated by a `# out=<path>` annotation the queue uses for exists-skip (the
queue never re-derives the filename from the flags).

Stages
------
  d2_clip     9 tasks x algo skrtrl-r16 x clip in {0.35, 0.7, 0.9} x seeds 0-4 x
              20000 steps, shadow on, svd_driver auto. Row order: clip 0.7 (most
              likely body figure) -> 0.9 -> 0.35, grouped as the outer loop so a
              truncated queue still finishes one full clip level across every
              task before starting the next. LR per (task, algo) comes from the
              D4 selection file (see --lr-json); the "none" clip level is not
              generated here -- it is read back from the D4 shadow-on logs
              (R2_EXPERIMENT_PLAN.md D2 / sec. 2).
  d2_noreset  TS tasks (henon, mackeyglass, lorenz, sunspot, laser) x
              skrtrl-r16 x --washout 0 x seeds 0-2 x 20000 steps, shadow on,
              default (no) spectral clip.

Usage
-----
  python make_r2_d2_jobs.py --stage clip --out jobs/d2_clip.txt
  python make_r2_d2_jobs.py --stage noreset --out jobs/d2_noreset.txt
  python make_r2_d2_jobs.py --stage all
  python make_r2_d2_jobs.py --status
"""
import argparse
import json
import os

from make_r2_jobs import (PY_M3, N_HID, BATCH, LOG_EVERY, TS_TASKS, REAL_TASKS,
                          WASHOUT, HORIZON, lr_tag)

# --- protocol constants (R2_EXPERIMENT_PLAN.md D2 / sec. 2 and 4) ---
TASKS_D2 = ["adding", "copy", "anbn", "rotation",
            "henon", "mackeyglass", "lorenz", "sunspot", "laser"]
ALGO_D2 = "skrtrl-r16"
CLIP_LEVELS = [0.7, 0.9, 0.35]   # row order: mid level first (most likely body figure)
SEEDS_CLIP = [0, 1, 2, 3, 4]
STEPS_CLIP = 20000

NORESET_TASKS = ["henon", "mackeyglass", "lorenz", "sunspot", "laser"]
SEEDS_NORESET = [0, 1, 2]
STEPS_NORESET = 20000

MS_PER_STEP = 30.0   # measured: n=64 skrtrl-r16, shadow on, svd_driver auto (sec. 2)

DEFAULT_LR_JSON = "results/r2/D4_SELECT_stage2.json"
DEFAULT_LR = 1e-3


def load_lr_table(path):
    """Read D4_SELECT_stage2.json's `pairs` list into {(task, algo): lr}.

    Missing/unreadable file -> empty table (caller falls back to --default-lr).
    """
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    table = {}
    for p in d.get("pairs", []):
        sel = p.get("selected")
        if sel and sel.get("lr") is not None:
            table[(p["task"], p["algo"])] = float(sel["lr"])
    return table


def get_lr(table, task, algo, default_lr):
    return table.get((task, algo), default_lr)


def m3_job_d2(task, algo, seed, lr, steps, outdir, tag, svd_driver="auto", shadow=1,
             clip=None, washout=None, extra=""):
    """Same shape as make_r2_jobs.m3_job, with an optional washout override
    (m3_job hardcodes WASHOUT=200; D2's no-reset stage needs --washout 0)."""
    parts = [PY_M3,
             f"--task {task}", f"--algo {algo}", f"--seed {seed}",
             f"--lr {lr:g}", f"--steps {steps}",
             f"--n {N_HID}", f"--batch {BATCH}", f"--log_every {LOG_EVERY}",
             f"--shadow {shadow}", f"--svd_driver {svd_driver}"]
    if clip is not None:
        parts.append(f"--clip {clip:g}")
    if task in TS_TASKS:
        causal = 1 if task in REAL_TASKS else 0
        w = WASHOUT if washout is None else washout
        parts += [f"--horizon {HORIZON}", f"--causal {causal}", f"--washout {w}"]
    parts += [f"--outdir {outdir}", f"--tag {tag}"]
    if extra:
        parts.append(extra)
    out = f"{outdir}/{task}_{algo}_s{seed}_{tag}.json"
    return " ".join(parts) + f"  # out={out}"


def clip_tag(clip):
    return f"clip{clip:g}"


def stage_clip(lr_table, default_lr, lr_json_path):
    outdir = "results/r2/d2_clip"
    n_jobs = len(TASKS_D2) * len(CLIP_LEVELS) * len(SEEDS_CLIP)
    gpu_h = n_jobs * STEPS_CLIP * MS_PER_STEP / 1000.0 / 3600.0
    jobs = [f"# D2 clip stage: {len(TASKS_D2)} tasks x algo {ALGO_D2} x "
            f"{len(CLIP_LEVELS)} clip levels {CLIP_LEVELS} x {len(SEEDS_CLIP)} seeds x "
            f"{STEPS_CLIP} steps, shadow on, svd_driver auto -> {n_jobs} jobs",
            f"# estimated GPU-h at {MS_PER_STEP:.0f} ms/step (n=64 skrtrl-r16 shadow-on "
            f"auto, measured): {gpu_h:.1f}",
            f"# row order: clip {CLIP_LEVELS} (mid level first); 'none' clip level is "
            f"NOT generated here -- read back from the D4 shadow-on logs",
            f"# LR source: {lr_json_path}" + ("" if lr_table else
                " (file not found -- using --default-lr for every row; "
                "REGENERATE once D4 selection (D4_SELECT_stage2.json) lands)")]
    for clip in CLIP_LEVELS:
        for task in TASKS_D2:
            lr = get_lr(lr_table, task, ALGO_D2, default_lr)
            for seed in SEEDS_CLIP:
                jobs.append(m3_job_d2(task, ALGO_D2, seed, lr, STEPS_CLIP, outdir,
                                      clip_tag(clip), clip=clip))
    return jobs


def stage_noreset(lr_table, default_lr, lr_json_path):
    outdir = "results/r2/d2_noreset"
    n_jobs = len(NORESET_TASKS) * len(SEEDS_NORESET)
    gpu_h = n_jobs * STEPS_NORESET * MS_PER_STEP / 1000.0 / 3600.0
    jobs = [f"# D2 no-reset stage: {len(NORESET_TASKS)} TS tasks x algo {ALGO_D2} x "
            f"--washout 0 x {len(SEEDS_NORESET)} seeds x {STEPS_NORESET} steps, "
            f"shadow on, default (no) spectral clip -> {n_jobs} jobs",
            f"# estimated GPU-h at {MS_PER_STEP:.0f} ms/step (measured): {gpu_h:.1f}",
            f"# LR source: {lr_json_path}" + ("" if lr_table else
                " (file not found -- using --default-lr for every row; "
                "REGENERATE once D4 selection (D4_SELECT_stage2.json) lands)")]
    for task in NORESET_TASKS:
        lr = get_lr(lr_table, task, ALGO_D2, default_lr)
        for seed in SEEDS_NORESET:
            jobs.append(m3_job_d2(task, ALGO_D2, seed, lr, STEPS_NORESET, outdir,
                                  "noreset", clip=None, washout=0))
    return jobs


STAGES = {"clip": stage_clip, "noreset": stage_noreset}
DEFAULT_OUT = {"clip": "jobs/d2_clip.txt", "noreset": "jobs/d2_noreset.txt"}


def write_jobs(lines, out_path, limit=0):
    if limit:
        head = [l for l in lines if l.startswith("#")]
        body = [l for l in lines if not l.startswith("#")][:limit]
        lines = head + body
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    n = sum(1 for l in lines if not l.startswith("#"))
    print(f"wrote {out_path}: {n} jobs")


def status(lr_table, default_lr, lr_json_path):
    for name, fn in STAGES.items():
        lines = fn(lr_table, default_lr, lr_json_path)
        body = [l for l in lines if not l.startswith("#")]
        expected = len(body)
        present = 0
        missing_examples = []
        for l in body:
            marker = "# out="
            i = l.rfind(marker)
            if i < 0:
                continue
            out = l[i + len(marker):].strip()
            if os.path.exists(out):
                present += 1
            elif len(missing_examples) < 3:
                missing_examples.append(out)
        print(f"[{name}] expected={expected} present={present} "
              f"missing={expected - present}")
        for ex in missing_examples:
            print(f"    e.g. missing: {ex}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["clip", "noreset", "all"], default="all")
    ap.add_argument("--out", default=None, help="output path (default per stage)")
    ap.add_argument("--lr-json", default=DEFAULT_LR_JSON,
                    help="D4_SELECT_stage2.json; keys (task, algo) -> selected lr")
    ap.add_argument("--default-lr", type=float, default=DEFAULT_LR,
                    help="used per (task, algo) row when --lr-json is missing or "
                         "has no selection for that pair yet")
    ap.add_argument("--limit", type=int, default=0,
                    help="keep only the first N real jobs per stage (smoke tests)")
    ap.add_argument("--status", action="store_true",
                    help="print expected/present job counts and exit (no write)")
    args = ap.parse_args()

    lr_table = load_lr_table(args.lr_json)

    if args.status:
        status(lr_table, args.default_lr, args.lr_json)
        return

    stages = ["clip", "noreset"] if args.stage == "all" else [args.stage]
    for name in stages:
        lines = STAGES[name](lr_table, args.default_lr, args.lr_json)
        out_path = args.out if (args.out and len(stages) == 1) else DEFAULT_OUT[name]
        write_jobs(lines, out_path, args.limit)


if __name__ == "__main__":
    main()
