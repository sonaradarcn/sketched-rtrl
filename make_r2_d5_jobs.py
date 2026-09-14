"""Generate R2 plan D5 (gated cells) job files for run_r2_queue.ps1 / .sh.

Same line format as make_r2_jobs.py / make_r2_d2_jobs.py: bare script name plus
flags, terminated by a `# out=<path>` annotation the queue uses for exists-skip
(authoritative -- the queue never re-derives the filename from the flags).

Grid (R2 plan D5 / paper/revise_r2/GRU_EXTENSION.tex sec. 8)
-----------------------------------------------------------
  cells   gru  (primary; append width omega = 2n, p_cell = 3 p_gate)
          lstm (secondary, emitted AFTER every gru row so a truncated queue
                still delivers the complete GRU table -- the LSTM block is
                droppable)
  tasks   rotation, anbn (episodic) + henon, sunspot (time series)
  algos   exact, snap1, skrtrl-r4, skrtrl-r16, skrtrl-r32, rflo, uoro
  seeds   0, 1, 2      steps 20000      shadow 1      svd_driver auto

  KF-RTRL is deliberately ABSENT: its Kronecker factorisation rests on
  A_t = D_t W and a rank-1 immediate Jacobian, neither of which holds for a
  gated cell (a GRU's immediate Jacobian is not even block diagonal), so
  run_m3.py refuses `--cell gru|lstm --algo kfrtrl` and D5 reports it as
  not-applicable.  TBPTT is absent for a different reason: it is a D4 fairness
  baseline, not part of the D5 estimator comparison, and `--algo tbptt_nnrnn`
  is tanh-only by construction.

Output names carry the cell (`<task>_<cell>_<algo>_s<seed>_<tag>.json`), so a
D5 run can never overwrite a tanh result; `--cell tanh` keeps the historical
name, which is why every existing job list stays valid.

Usage
-----
  python make_r2_d5_jobs.py                      # -> jobs/d5_gated.txt
  python make_r2_d5_jobs.py --out jobs/x.txt --limit 4
  python make_r2_d5_jobs.py --status
  python make_r2_d5_jobs.py --cells gru          # gru block only
"""
import argparse
import os

from make_r2_jobs import (PY_M3, N_HID, BATCH, LOG_EVERY, TS_TASKS, REAL_TASKS,
                          WASHOUT, HORIZON)
from make_r2_d2_jobs import load_lr_table, get_lr, DEFAULT_LR_JSON, DEFAULT_LR

# --- protocol constants ---------------------------------------------------
CELLS_D5 = ["gru", "lstm"]          # order is the row order: gru first
TASKS_D5 = ["rotation", "anbn", "henon", "sunspot"]
ALGOS_D5 = ["exact", "snap1", "skrtrl-r4", "skrtrl-r16", "skrtrl-r32",
            "rflo", "uoro"]
SEEDS_D5 = [0, 1, 2]
STEPS_D5 = 20000
TAG_D5 = "d5"
OUTDIR_D5 = "results/r2/d5_gated"

# ms/step MEASURED for the GRU on an RTX 3080 (n=64, batch 8, shadow on,
# svd_driver auto, 200 steps of `rotation`) WHILE two other queues were running
# on the same GPU.  These are loaded figures, so the GPU-h below is an upper
# bound; the same measurement puts the tanh cell at 71.7 ms/step for
# skrtrl-r16, against the established uncontended reference of 30 ms/step
# (make_r2_d2_jobs.MS_PER_STEP), hence CONTENTION_FACTOR -- the header quotes
# both the loaded total and the de-contended one.
MS_PER_STEP = {
    "exact": 13.0,
    "snap1": 16.0,
    "skrtrl-r4": 92.0,
    "skrtrl-r16": 97.0,
    "skrtrl-r32": 115.0,
    "rflo": 12.0,
    "uoro": 17.0,
}
MS_DEFAULT = 100.0
CONTENTION_FACTOR = 30.0 / 71.7      # uncontended / loaded, from the tanh r16 pair
# multiplier on top of MS_PER_STEP per cell: measured 97 -> 99 ms/step at r=16
# and 115 -> 116 at r=32 (the SVD shapes differ -- the GRU pre-projects an
# (n, 2n) append, the LSTM a (2n, n) one), so the LSTM is ~5% dearer per step.
# The real cost difference is memory, not time: peak 56 MB (tanh) -> 137 MB
# (gru) -> 325 MB (lstm) at n=64, batch 8, shadow on.
CELL_MS_MULT = {"gru": 1.0, "lstm": 1.05}


def m3_job_d5(cell, task, algo, seed, lr, steps=STEPS_D5, outdir=OUTDIR_D5,
              tag=TAG_D5, svd_driver="auto", shadow=1):
    parts = [PY_M3,
             f"--cell {cell}",
             f"--task {task}", f"--algo {algo}", f"--seed {seed}",
             f"--lr {lr:g}", f"--steps {steps}",
             f"--n {N_HID}", f"--batch {BATCH}", f"--log_every {LOG_EVERY}",
             f"--shadow {shadow}", f"--svd_driver {svd_driver}"]
    if task in TS_TASKS:
        causal = 1 if task in REAL_TASKS else 0
        parts += [f"--horizon {HORIZON}", f"--causal {causal}",
                  f"--washout {WASHOUT}"]
    parts += [f"--outdir {outdir}", f"--tag {tag}"]
    cellpart = "" if cell == "tanh" else f"_{cell}"
    out = f"{outdir}/{task}{cellpart}_{algo}_s{seed}_{tag}.json"
    return " ".join(parts) + f"  # out={out}"


def gpu_hours(cells):
    """(total_h, per-cell dict).  Sum over the grid of steps * ms/step."""
    per = {}
    for cell in cells:
        h = 0.0
        for _task in TASKS_D5:
            for algo in ALGOS_D5:
                ms = MS_PER_STEP.get(algo, MS_DEFAULT) * CELL_MS_MULT.get(cell, 1.0)
                h += len(SEEDS_D5) * STEPS_D5 * ms / 1000.0 / 3600.0
        per[cell] = h
    return sum(per.values()), per


def stage_d5(cells, lr_table, default_lr, lr_json_path):
    n_per_cell = len(TASKS_D5) * len(ALGOS_D5) * len(SEEDS_D5)
    total_h, per = gpu_hours(cells)
    lines = [
        f"# D5 gated-cell stage: {len(cells)} cells {cells} x {len(TASKS_D5)} tasks "
        f"{TASKS_D5} x {len(ALGOS_D5)} algos {ALGOS_D5} x {len(SEEDS_D5)} seeds x "
        f"{STEPS_D5} steps, shadow on, svd_driver auto "
        f"-> {n_per_cell} jobs per cell, {n_per_cell * len(cells)} total",
        "# KF-RTRL is NOT in the grid: not applicable to a gated cell (Kronecker "
        "factorisation needs A_t = D_t W and a rank-1, block-diagonal immediate "
        "Jacobian); run_m3.py refuses it for --cell gru|lstm",
        "# row order: every gru row first, then lstm -- the lstm block is the "
        "droppable one",
        "# estimated GPU-h from loaded-GPU ms/step (upper bound): "
        + ", ".join(f"{c} {per[c]:.1f}" for c in cells)
        + f"; total {total_h:.1f}",
        f"# same estimate de-contended (x{CONTENTION_FACTOR:.2f}, from the tanh "
        f"skrtrl-r16 loaded/uncontended pair 71.7/30 ms per step): "
        + ", ".join(f"{c} {per[c] * CONTENTION_FACTOR:.1f}" for c in cells)
        + f"; total {total_h * CONTENTION_FACTOR:.1f}",
        "# memory: n=64 batch 8 shadow on -> peak 137 MB (gru), 325 MB (lstm) "
        "vs 56 MB (tanh); --shadow_max_n is interpreted as a vanilla-cell width, "
        "so the shadow is admitted up to ~0.69 n_max (gru) / ~0.50 n_max (lstm)",
        f"# LR source: {lr_json_path}" + ("" if lr_table else
            " (file not found -- using --default-lr for every row; REGENERATE "
            "once D4 selection (D4_SELECT_stage2.json) lands.  NOTE: the D4 "
            "table was tuned on the TANH cell; treat it as a starting point and "
            "re-tune if the gated runs are lr-limited)"),
    ]
    for cell in cells:
        for task in TASKS_D5:
            for algo in ALGOS_D5:
                lr = get_lr(lr_table, task, algo, default_lr)
                for seed in SEEDS_D5:
                    lines.append(m3_job_d5(cell, task, algo, seed, lr))
    return lines


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


def status(cells, lr_table, default_lr, lr_json_path):
    body = [l for l in stage_d5(cells, lr_table, default_lr, lr_json_path)
            if not l.startswith("#")]
    present, missing = 0, []
    for l in body:
        i = l.rfind("# out=")
        if i < 0:
            continue
        out = l[i + len("# out="):].strip()
        if os.path.exists(out):
            present += 1
        elif len(missing) < 3:
            missing.append(out)
    print(f"[d5_gated] expected={len(body)} present={present} "
          f"missing={len(body) - present}")
    for ex in missing:
        print(f"    e.g. missing: {ex}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="jobs/d5_gated.txt")
    ap.add_argument("--cells", default=",".join(CELLS_D5),
                    help="comma-separated subset of gru,lstm (order = row order)")
    ap.add_argument("--lr-json", default=DEFAULT_LR_JSON)
    ap.add_argument("--default-lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--limit", type=int, default=0,
                    help="keep only the first N real jobs (smoke tests)")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    bad = [c for c in cells if c not in CELLS_D5]
    if bad:
        raise SystemExit(f"unknown cell(s) {bad}; have {CELLS_D5}")
    lr_table = load_lr_table(args.lr_json)
    if args.status:
        status(cells, lr_table, args.default_lr, args.lr_json)
        return
    write_jobs(stage_d5(cells, lr_table, args.default_lr, args.lr_json),
               args.out, args.limit)


if __name__ == "__main__":
    main()
