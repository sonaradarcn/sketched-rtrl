"""Generate the D1 (residual-spectrum, R3-1) job list for the R2 queue.

Coverage (R2_EXPERIMENT_PLAN.md, D1):
  (a) every non-RL task at n=64, seeds 0-2, 20 000 steps, batch 8
  (b) width sweep on the four diagnostic tasks, n in {256, 512, 128}, seeds 0-2,
      batch 8 (batch 4 at n=512, where the exact shadow needs ~4.2 GB).  The sweep is
      emitted in the order 64 / 256 / 512 / 128 so that the widths that decide the
      "does the required rank grow with n" figure land first and n=128 (the cheapest
      interpolation point) fills in last.
The T-maze RL spectrum is NOT included here: it needs the shadow + SnAp-1 trace added to
the RL core first, and is queued separately once that lands.

Every line is a complete command and ends with `# out=<json path>` so the queue can do
exists-skip without importing anything.  This script only writes the list; it launches
nothing.

Usage:
  python make_r2_d1_jobs.py                # -> jobs/d1_spectrum.txt (bare script lines,
                                           #    same convention as the other jobs/*.txt)
  python make_r2_d1_jobs.py --py python    # prepend an interpreter to every line
  python make_r2_d1_jobs.py --status       # which outputs already exist
"""
import argparse, os

ALL_TASKS = ["adding", "copy", "anbn", "rotation",          # diagnostic
             "henon", "mackeyglass", "lorenz",              # chaotic
             "sunspot", "laser"]                            # real
DIAGNOSTIC = ["adding", "copy", "anbn", "rotation"]
TS_TASKS = {"henon", "mackeyglass", "lorenz", "sunspot", "laser"}
SEEDS = [0, 1, 2]
WIDTHS = [256, 512, 128]      # queue order: informative widths first, 128 fills in

# measured on one RTX 3080 (exact RTRL + passive SnAp-1 tracker): training cost per step
# and the cost of one age-stratified checkpoint (2 spectra x batch lanes).
MS_PER_STEP = {64: 4.8, 128: 5.0, 256: 10.0, 512: 39.0}
CKPT_S = {64: 0.25, 128: 0.45, 256: 0.40, 512: 0.55}


def batch_for(n):
    return 4 if n >= 512 else 8


def est_hours(n, steps, n_ckpt):
    return (steps * MS_PER_STEP[n] / 1000.0 + n_ckpt * CKPT_S[n]) / 3600.0


def build(args):
    fracs = args.ckpt_fracs
    n_stage = len([f for f in fracs.split(",") if f.strip()])
    # --ckpt_align age puts 5 age-stratified checkpoints in each training stage
    n_ckpt = n_stage * (5 if args.ckpt_align == "age" else 1)
    jobs = []
    grid = [(t, 64) for t in ALL_TASKS] if not args.widths_only else []
    if not args.base_only:
        grid += [(t, n) for n in WIDTHS for t in DIAGNOSTIC]
    for task, n in grid:
        for seed in SEEDS:
            b = batch_for(n)
            out = f"{args.outdir}/{task}_n{n}_s{seed}.json"
            cmd = [args.script,
                   f"--task {task}", f"--n {n}", f"--seed {seed}",
                   f"--steps {args.steps}", f"--batch {b}", f"--lr {args.lr}",
                   f"--ckpt_fracs {fracs}", f"--ckpt_align {args.ckpt_align}",
                   f"--outdir {args.outdir}"]
            if task in TS_TASKS:
                cmd += [f"--horizon {args.horizon}", f"--causal {args.causal}",
                        f"--washout {args.washout}"]
            if args.svd_driver:
                cmd += [f"--svd_driver {args.svd_driver}"]
            line = " ".join(([args.py] if args.py else []) + cmd)
            jobs.append({"task": task, "n": n, "seed": seed, "batch": b,
                         "line": f"{line}  # out={out}", "out": out,
                         "hours": est_hours(n, args.steps, n_ckpt)})
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default="run_e1_spectrum.py")
    ap.add_argument("--py", default="", help='interpreter prefix; default "" emits bare '
                    'script lines, matching the other R2 job files in jobs/')
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", default="1e-3")
    ap.add_argument("--ckpt_fracs", default="0.01,0.05,0.25,0.5,0.75,1.0")
    ap.add_argument("--ckpt_align", default="age", choices=["age", "mid", "none"])
    ap.add_argument("--outdir", default="results/r2/d1_spectrum")
    ap.add_argument("--out", default="jobs/d1_spectrum.txt")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--causal", type=int, default=0)
    ap.add_argument("--washout", type=int, default=200)
    ap.add_argument("--svd_driver", default="", help="passed through for queue uniformity only")
    ap.add_argument("--base_only", action="store_true", help="only the n=64 all-task block")
    ap.add_argument("--widths_only", action="store_true", help="only the width sweep block")
    ap.add_argument("--status", action="store_true", help="report existing outputs, write nothing")
    args = ap.parse_args()

    jobs = build(args)
    total_h = sum(j["hours"] for j in jobs)
    by_n = {}
    for j in jobs:
        d = by_n.setdefault(j["n"], {"count": 0, "hours": 0.0, "batch": j["batch"]})
        d["count"] += 1
        d["hours"] += j["hours"]

    if args.status:
        done = [j for j in jobs if os.path.exists(j["out"])]
        print(f"{len(done)}/{len(jobs)} outputs already exist "
              f"({sum(j['hours'] for j in jobs if j not in done):.2f} GPU-h remaining)")
        for j in jobs:
            if not os.path.exists(j["out"]):
                print("  todo:", j["out"])
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="\n") as f:
        f.write("# R2 D1 residual-spectrum jobs (R_t = J_t - S_t), generated by make_r2_d1_jobs.py\n")
        f.write(f"# {len(jobs)} jobs, {args.steps} steps each, estimated {total_h:.2f} GPU-h total "
                f"(~{total_h / 2:.2f} h wall-clock on 2 GPUs)\n")
        for n in sorted(by_n):
            d = by_n[n]
            f.write(f"#   n={n:<4} batch {d['batch']}  {d['count']:3d} jobs  {d['hours']:.2f} GPU-h\n")
        f.write("# T-maze RL spectrum not included (needs shadow+trace in the RL core first).\n")
        n_stage = len([f_ for f_ in args.ckpt_fracs.split(",") if f_.strip()])
        per_stage = 5 if args.ckpt_align == "age" else 1
        f.write(f"# ckpt_align={args.ckpt_align}: {n_stage * per_stage} checkpoints per run "
                f"({n_stage} training stages x {per_stage} ages within the reset interval)\n")
        f.write("# Each line ends with '# out=<json>' for exists-skip; n=512 needs ~4.2 GB.\n")
        for j in jobs:
            f.write(j["line"] + "\n")
    print(f"wrote {args.out}: {len(jobs)} jobs, {total_h:.2f} GPU-h estimated")
    for n in sorted(by_n):
        d = by_n[n]
        print(f"  n={n:<4} batch {d['batch']}  {d['count']:3d} jobs  {d['hours']:.2f} GPU-h")


if __name__ == "__main__":
    main()
