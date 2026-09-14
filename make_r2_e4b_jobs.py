"""Generate the D4b (E4b) common-trajectory job list for the R2 queue.

Coverage (R2_EXPERIMENT_PLAN.md, D4 addition: "add a common-trajectory tracking
comparison -- all estimators passively tracking one exact-trained trajectory per task,
3 seeds"):

    9 non-RL tasks  x  n=64  x  seeds {0,1,2}  x  20 000 steps  x  batch 8
    trackers: skrtrl-r4, skrtrl-r16, skrtrl-r32, snap1, uoro, kfrtrl, rflo
              (TBPTT is not an online per-step estimator -- see run_e4b_common.py)

One line per job, complete command, ending in `# out=<json path>` so run_r2_queue.ps1
can exists-skip without importing anything.  This script writes the list only; it
launches nothing.

Cost model (measured)
---------------------
One RTX 3080, all 7 trackers, cert_every 1, svd_driver auto, n=64, batch 8, 3 000-step
runs, driver 560.94, **with the other GPU running the R2 queue** (the realistic regime:
the D4b list itself is run by two concurrent workers):

    rotation  n=64  ->  222.1 ms/step, peak 63 MB, 18 000 SVDs, no driver fallback
    henon     n=64  ->  270.3 ms/step, peak 59 MB, 18 000 SVDs, no driver fallback

The same rotation run on an otherwise idle machine took 158.4 ms/step, so contention
costs ~40 %.  The estimate below uses the contended numbers (conservative); the header
also prints the idle-machine figure.

Attribution (rotation, 600 steps, idle machine):
        all 7 trackers                    140.8 ms/step
        cert_every 10                     125.5     -> instrumentation ~16 ms / labelled step
        rng_isolate 0                     132.7     -> private RNG streams ~8 ms/step
        skrtrl-r16 only                    49.0
        snap1+uoro+kfrtrl+rflo (no SVD)    11.1     -> the 3 SK-RTRL SVD kernels dominate

Model for the tasks that were not timed:

    ms/step = BASE_MS + INSTR_MS * label_density / cert_every

`label_density` is the fraction of steps at which the task emits a target (adding and
copy are sparse, everything else is 1.0).  The per-task variation with p = n + n_in + 1
is not resolvable against the run-to-run contention spread (rotation p=74 came out FASTER
than henon p=66), so BASE_MS is the mean of the two dense-label measurements and the
per-task numbers below carry roughly a +/- 20 % uncertainty.

Budget levers: the cost is ~90 % the three SK-RTRL SVD kernels, so `--cert_every 10`
saves only ~7 %; halving `--steps` halves the cost, and dropping one SK-RTRL rank from
`--trackers` saves ~1/3 of the SVD time (~10 GPU-h).

Usage
-----
  python make_r2_e4b_jobs.py                                  # -> jobs/e4b_common.txt
  python make_r2_e4b_jobs.py --py D:/Anaconda/envs/multilingual_lora/python.exe
  python make_r2_e4b_jobs.py --status                         # which outputs exist
"""
import argparse, os

SCRIPT = "run_e4b_common.py"

ALL_TASKS = ["adding", "copy", "anbn", "rotation",          # diagnostic
             "henon", "mackeyglass", "lorenz",              # chaotic
             "sunspot", "laser"]                            # real
TS_TASKS = {"henon", "mackeyglass", "lorenz", "sunspot", "laser"}
REAL_TASKS = {"sunspot", "laser"}          # causal normalization, no future leakage
SEEDS = [0, 1, 2]
N_HID = 64
BATCH = 8
WASHOUT = 200
HORIZON = 1
TRACKERS = "skrtrl-r4,skrtrl-r16,skrtrl-r32,snap1,uoro,kfrtrl,rflo"

# task input width -> p = n + n_in + 1, P = n * p (drives the SK-RTRL right-factor cost)
N_IN = {"adding": 2, "copy": 10, "anbn": 3, "rotation": 9,
        "henon": 1, "mackeyglass": 1, "lorenz": 1, "sunspot": 1, "laser": 1}
# fraction of steps that carry a target (delta_t, hence the whole instrumentation)
LABEL_DENSITY = {"adding": 1.0 / 100, "copy": 5.0 / 50, "anbn": 1.0, "rotation": 1.0,
                 "henon": 1.0, "mackeyglass": 1.0, "lorenz": 1.0,
                 "sunspot": 1.0, "laser": 1.0}

# GPU ms/step measured at n=64, batch 8, cert_every 1, other GPU busy (see docstring)
MS_MEASURED = {"rotation": 222.1, "henon": 270.3}
BASE_MS = 230.0            # dense-label tracker steps at n=64, instrumentation excluded
BASE_MS_IDLE = 142.4       # same, measured on an otherwise idle machine
INSTR_MS = 16.0            # per labelled+instrumented step (ghat, ||E||_F, counters)
PEAK_MB = 63               # measured peak allocation at n=64, batch 8, 7 trackers


def p_of(task, n):
    return n + N_IN[task] + 1


def ms_per_step(task, n, cert_every, base=BASE_MS):
    """Measured where available, else base + instrumentation; +/- 20 % (see docstring)."""
    if task in MS_MEASURED and n == N_HID and cert_every == 1 and base == BASE_MS:
        return MS_MEASURED[task]
    dens = LABEL_DENSITY[task] / max(cert_every, 1)
    return base + INSTR_MS * dens


def build(args):
    jobs = []
    for task in ALL_TASKS:
        for seed in SEEDS:
            out = f"{args.outdir}/{task}_n{args.n}_s{seed}.json"
            cmd = [SCRIPT,
                   f"--task {task}", f"--n {args.n}", f"--seed {seed}",
                   f"--steps {args.steps}", f"--batch {args.batch}", f"--lr {args.lr}",
                   f"--log_every {args.log_every}",
                   f"--trackers {args.trackers}",
                   f"--svd_driver {args.svd_driver}",
                   f"--cert_every {args.cert_every}"]
            if args.clip:
                cmd.append(f"--clip {args.clip:g}")
            if task in TS_TASKS:
                causal = 1 if task in REAL_TASKS else 0
                cmd += [f"--horizon {HORIZON}", f"--causal {causal}",
                        f"--washout {WASHOUT}"]
            cmd += [f"--outdir {args.outdir}"]
            if args.tag:
                cmd.append(f"--tag {args.tag}")
                out = f"{args.outdir}/{task}_n{args.n}_s{seed}_{args.tag}.json"
            line = " ".join(([args.py] if args.py else []) + cmd)
            hours = args.steps * ms_per_step(task, args.n, args.cert_every) / 1000.0 / 3600.0
            jobs.append({"task": task, "seed": seed, "out": out, "hours": hours,
                         "line": f"{line}  # out={out}"})
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--py", default="", help='interpreter prefix; "" emits bare script lines '
                                             "(run_r2_queue.ps1 prepends -Python)")
    ap.add_argument("--n", type=int, default=N_HID)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", default="1e-3")
    ap.add_argument("--log_every", type=int, default=250)
    ap.add_argument("--trackers", default=TRACKERS)
    ap.add_argument("--svd_driver", default="auto", choices=["auto", "gesvd"])
    ap.add_argument("--cert_every", type=int, default=1,
                    help="stride of the per-step instrumentation; >1 cuts ~16 ms/step")
    ap.add_argument("--clip", type=float, default=0.0)
    ap.add_argument("--outdir", default="results/r2/e4b")
    ap.add_argument("--out", default="jobs/e4b_common.txt")
    ap.add_argument("--tag", default="")
    ap.add_argument("--limit", type=int, default=0, help="keep only the first N jobs (smoke)")
    ap.add_argument("--status", action="store_true", help="report existing outputs, write nothing")
    args = ap.parse_args()

    jobs = build(args)
    if args.limit:
        jobs = jobs[:args.limit]
    total_h = sum(j["hours"] for j in jobs)
    idle_h = sum(args.steps * ms_per_step(j["task"], args.n, args.cert_every,
                                          base=BASE_MS_IDLE) / 3.6e6 for j in jobs)
    by_task = {}
    for j in jobs:
        d = by_task.setdefault(j["task"], {"count": 0, "hours": 0.0})
        d["count"] += 1
        d["hours"] += j["hours"]

    if args.status:
        todo = [j for j in jobs if not os.path.exists(j["out"])]
        print(f"{len(jobs) - len(todo)}/{len(jobs)} outputs already exist "
              f"({sum(j['hours'] for j in todo):.2f} GPU-h remaining)")
        for j in todo:
            print("  todo:", j["out"])
        return

    d = os.path.dirname(args.out)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("# R2 D4b common-trajectory estimator comparison, generated by "
                "make_r2_e4b_jobs.py\n")
        f.write(f"# exact-RTRL trainer + passive trackers [{args.trackers}]; TBPTT n/a\n")
        f.write(f"# {len(jobs)} jobs = {len(ALL_TASKS)} tasks x n={args.n} x seeds "
                f"{SEEDS} x {args.steps} steps, batch {args.batch}, "
                f"svd_driver {args.svd_driver}, cert_every {args.cert_every}\n")
        f.write(f"# estimated {total_h:.2f} GPU-h total (~{total_h / 2:.2f} h wall-clock on "
                f"2 GPUs); cost is dominated by the 3 SK-RTRL SVD kernels\n")
        f.write(f"# measured n=64 rates: rotation 222.1, henon 270.3 ms/step with the "
                f"other GPU busy; on an idle machine the total is ~{idle_h:.1f} GPU-h "
                f"(rotation 158.4 ms/step). Per-task rows +/- 20 %.\n")
        for t in ALL_TASKS:
            b = by_task.get(t)
            if b:
                f.write(f"#   {t:<12} {b['count']} jobs  {b['hours']:.2f} GPU-h  "
                        f"({ms_per_step(t, args.n, args.cert_every):.0f} ms/step est)\n")
        f.write(f"# Each line ends with '# out=<json>' for exists-skip; peak ~{PEAK_MB} MB "
                f"at n={args.n}, batch {args.batch}.\n")
        for j in jobs:
            f.write(j["line"] + "\n")
    print(f"wrote {args.out}: {len(jobs)} jobs, {total_h:.2f} GPU-h estimated "
          f"(~{total_h / 2:.2f} h on 2 GPUs; {idle_h:.2f} GPU-h on an idle machine)")
    for t in ALL_TASKS:
        b = by_task.get(t)
        if b:
            print(f"  {t:<12} {b['count']} jobs  {b['hours']:.2f} GPU-h  "
                  f"({ms_per_step(t, args.n, args.cert_every):.0f} ms/step est)")


if __name__ == "__main__":
    main()
