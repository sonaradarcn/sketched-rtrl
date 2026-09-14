"""Generate the RL job list for the R2 queue (plan D1/D2, RL part).

Grid (fixed after the external plan review)
-------------------------------------------
  SK-RTRL r=16 x tmaze{10,20,40} x clip{0.0, 0.5} x seeds{0,1,2}   18 runs
  SnAp-1        x tmaze{10,20,40} x clip 0.0      x seeds{0,1,2}     9 runs
all with --shadow 1, n=64, batch 16, lr 3e-4 (the archived R1 RL protocol).
tmaze40 is in on purpose: it is the one corridor where R1 reports no advantage,
so it cannot be missing from the R2 evidence.

Order = priority, and the queue walks the list in order on every worker (it
splits by line index modulo the worker count), so a tail that is cut still
leaves a balanced grid:
  1. SK-RTRL clip 0.0, corridors 10 / 20 / 40   (the R1 setting: no clipping)
  2. SK-RTRL clip 0.5, corridors 10 / 20 / 40   (the informative-certificate arm)
  3. SnAp-1  clip 0.0, corridors 10 / 20 / 40   (residual reference, droppable)

Step budgets (--steps_map, basis printed into the job file)
----------------------------------------------------------
From the archived R1 runs in results/m5iso (n=64, batch 16, lr 3e-4):
  tmaze10  60 000 steps: exact reaches success 1.00, SK-RTRL-r16 peaks at 1.00
           by 20k and decays to 0.67 by 60k -- the whole rise-and-decay is inside
           60k, so 60k is the smallest budget that still shows learning.
  tmaze20 100 000 steps: SK-RTRL-r16 goes 0.40 @40k -> 0.76 @60k -> 0.67 @100k,
           i.e. the transition sits inside 60k but only settles by 100k; R1 used
           100k, so 100k keeps the D1/D2 numbers comparable to Table 9.
  tmaze40  60 000 steps (R1 used 150 000): at 150k EVERY estimator is still at
           chance (exact 0.49, SK-RTRL-r16 0.52, SnAp-1 0.49), so no additional
           learning regime exists to capture; 60k characterises the same
           non-learning regime at 40 % of the cost.  Raise it with
           --steps_map 40:150000 if the campaign wants R1's exact budget.

Line format is the one run_r2_queue.ps1 consumes: a full command line ending in
`# out=<json path>`, which is authoritative for exists-skip (run_m5.py also skips
by itself when the output file exists).  A line may start with the bare .py
script (--py ""), with "python", or with an explicit interpreter.

  python make_r2_rl_jobs.py                          # -> results/r2/rl_jobs.txt
  python make_r2_rl_jobs.py --out -                  # stdout
  python make_r2_rl_jobs.py --steps 60000            # uniform budget instead
  python make_r2_rl_jobs.py --status                 # which outputs exist already

This script launches nothing.

NOTE on the queue timeout: run_r2_queue.ps1 defaults to -TimeoutSec 10800 (3 h).
With --svd_driver auto (the default here, as in every other R2 job file; the
driver-equivalence gate is A1's) the longest job is ~2.3 h, which fits but has
little margin under GPU contention -- pass -TimeoutSec 18000.  With
--svd_driver gesvd the SK-RTRL jobs take 2.5x longer and WILL time out.
"""
import argparse
import os
import sys

# (algo, clip) arms in priority order
ARMS = [("skrtrl-r16", 0.0), ("skrtrl-r16", 0.5), ("snap1", 0.0)]
CORRIDORS = [10, 20, 40]
SEEDS = [0, 1, 2]
STEPS_MAP = {10: 60000, 20: 100000, 40: 60000}

# ms/step measured on this machine (RTX 3080, n=64, batch 16, shadow on; the
# per-step cost does not depend on the corridor length).  Only used for the
# GPU-hour estimate in the job-file header.
MS_PER_STEP = {
    "skrtrl-r16": {"gesvd": 206.0, "auto": 83.0},
    "snap1": {"gesvd": 15.9, "auto": 15.9},
}


def parse_map(spec, default):
    if not spec:
        return dict(default)
    out = {}
    for tok in str(spec).split(","):
        tok = tok.strip()
        if tok:
            k, v = tok.split(":")
            out[int(k)] = int(v)

    return out


def outdir_for(base, clip):
    if clip <= 0:
        return base
    return base + "_clip%s" % ("%g" % clip).replace(".", "")


def build(args):
    steps_map = parse_map(args.steps_map, STEPS_MAP)
    arms = [(a, c) for a, c in ARMS if a in
            [x.strip() for x in args.algos.split(",")]] if args.algos else ARMS
    corridors = [int(v) for v in args.lengths.split(",") if v.strip()]
    seeds = [int(v) for v in args.seeds.split(",") if v.strip()]
    jobs = []
    for algo, clip in arms:
        for ln in corridors:
            steps = args.steps or steps_map.get(ln, 60000)
            outdir = outdir_for(args.outdir, clip)
            for seed in seeds:
                out = f"{outdir}/tmaze{ln}_{algo}_s{seed}.json"
                cmd = [args.script,
                       "--env_len", str(ln),
                       "--algo", algo,
                       "--seed", str(seed),
                       "--n", str(args.n),
                       "--steps", str(steps),
                       "--batch", str(args.batch),
                       "--lr", str(args.lr),
                       "--log_every", str(args.log_every),
                       "--shadow", "1",
                       "--spectrum_ckpts", args.spectrum_ckpts,
                       "--cert_warmup", str(args.cert_warmup),
                       "--cert_stages", str(args.cert_stages),
                       "--clip", ("%g" % clip),
                       "--svd_driver", args.svd_driver,
                       "--outdir", outdir]
                if args.device != "cuda":
                    cmd += ["--device", args.device]
                line = " ".join(cmd)
                if args.py:
                    line = f"{args.py} {line}"
                jobs.append({"line": f"{line}  # out={out}", "out": out,
                             "algo": algo, "clip": clip, "env_len": ln,
                             "seed": seed, "steps": steps})
    return jobs


def gpu_hours(jobs, driver):
    total, unknown = 0.0, []
    per_arm = {}
    for j in jobs:
        ms = MS_PER_STEP.get(j["algo"], {}).get(driver)
        if ms is None:
            unknown.append(j["algo"])
            continue
        h = ms * j["steps"] / 3.6e6
        total += h
        key = (j["algo"], j["clip"], j["env_len"], j["steps"])
        per_arm[key] = per_arm.get(key, 0.0) + h
    return total, sorted(set(unknown)), per_arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="10,20,40")
    ap.add_argument("--algos", default="",
                    help="restrict the arms to these algos (default: all)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=0,
                    help="uniform step budget; 0 = per-corridor --steps_map")
    ap.add_argument("--steps_map", default="",
                    help='per-corridor budgets, e.g. "10:60000,20:100000,40:60000"')
    ap.add_argument("--n", type=int, default=64)           # archived R1 RL protocol
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", default="3e-4")
    ap.add_argument("--log_every", type=int, default=500)
    # same checkpoint set as make_r2_d1_jobs.py (plan D1: 1, 5, 25, 50, 75, 100 %)
    ap.add_argument("--spectrum_ckpts", default="0.01,0.05,0.25,0.5,0.75,1.0")
    ap.add_argument("--cert_warmup", type=int, default=200)
    ap.add_argument("--cert_stages", type=int, default=3)
    ap.add_argument("--svd_driver", choices=["gesvd", "auto"], default="auto")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default="results/r2/rl")
    ap.add_argument("--script", default="run_m5.py")
    ap.add_argument("--py", default="python",
                    help='interpreter prefix; "" emits bare script lines')
    ap.add_argument("--out", default="results/r2/rl_jobs.txt",
                    help='job file to write; "-" for stdout')
    ap.add_argument("--status", action="store_true",
                    help="only report which outputs already exist")
    args = ap.parse_args()

    jobs = build(args)
    if args.status:
        done = 0
        for j in jobs:
            ok = os.path.exists(j["out"])
            done += int(ok)
            print(("done " if ok else "TODO ") + j["out"])
        print(f"{done}/{len(jobs)} outputs present")
        return

    gph, unknown, per_arm = gpu_hours(jobs, args.svd_driver)
    budgets = ", ".join(f"tmaze{ln}={s}"
                        for ln, s in sorted({(j["env_len"], j["steps"])
                                             for j in jobs}))
    head = [
        "# R2 RL jobs (T-maze residual spectrum + certificate, plan D1/D2),",
        "# generated by make_r2_rl_jobs.py -- see its docstring for the step-budget basis.",
        f"# {len(jobs)} jobs: SK-RTRL-r16 x clip{{0,0.5}} + SnAp-1 x clip0, "
        f"corridors {args.lengths}, seeds {args.seeds}, shadow on, "
        f"n={args.n}, batch={args.batch}, lr={args.lr}, svd_driver {args.svd_driver}",
        f"# step budgets ({budgets}) from the archived R1 runs in results/m5iso:",
        "#   tmaze10 60k  -- exact hits success 1.00 and SK-RTRL-r16 peaks 1.00@20k "
        "then decays to 0.67@60k: the whole trajectory is inside 60k",
        "#   tmaze20 100k -- SK-RTRL-r16 0.40@40k -> 0.76@60k -> 0.67@100k: the "
        "transition needs >60k to settle, and R1 used 100k",
        "#   tmaze40 60k  -- R1 ran 150k and every estimator stayed at chance "
        "(exact 0.49 / SK-RTRL 0.52 / SnAp-1 0.49), so longer buys no new regime",
        f"# estimated {gph:.1f} GPU-h total (~{gph / 2:.1f} h wall-clock on 2 GPUs)"
        + (f"  [no timing model for {', '.join(unknown)}]" if unknown else ""),
        "# order = priority (the queue walks the list in order on every worker); "
        "the SnAp-1 tail is the droppable part",
        "# each line ends with '# out=<json>' for exists-skip; peak GPU memory ~80 MB/run",
        f"# outputs: {args.outdir}/ (clip 0) and "
        f"{outdir_for(args.outdir, 0.5)}/ (clip 0.5)",
        "# queue: pass -TimeoutSec 18000 (the longest job is ~2.3 h with svd_driver auto)",
    ]
    text = "\n".join(head + [j["line"] for j in jobs]) + "\n"

    if args.out == "-":
        sys.stdout.write(text)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="\n") as f:
            f.write(text)
        print(f"wrote {len(jobs)} jobs -> {args.out} ({gph:.1f} GPU-h estimated)")
        for key in sorted(per_arm):
            algo, clip, ln, steps = key
            print(f"  {algo:11s} clip {clip:<4g} tmaze{ln:<3d} {steps:6d} steps "
                  f"x {len(SEEDS)} seeds = {per_arm[key]:5.2f} GPU-h")


if __name__ == "__main__":
    main()
