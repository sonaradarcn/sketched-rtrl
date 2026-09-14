"""Generate R2 plan D3 (adaptive-rank sensitivity) job files for run_r2_queue.ps1.

Same line format as make_r2_jobs.py / jobs/d4_tune.txt: bare script name + flags,
terminated by a `# out=<path>` annotation the queue uses for exists-skip. Jobs run
run_adaptive.py, whose output path is
  <outdir>/<task>_<fixed{R}|adaptive-{ctrl}>_s<seed>_<tag>.json
(see run_adaptive.py's `base`/`out` construction) -- reproduced here as
adapt_out_path() so this generator and the runner cannot disagree.

Stages (R2_EXPERIMENT_PLAN.md D3, revised in sec. 4: saturation diagnosis first,
then a narrower one-at-a-time sweep)
------------------------------------
  d3_diag  rotation, anbn (diagnostic tasks, no washout/horizon) x
           5 kernel variants (A-E, all --ctrl eta, one-at-a-time around the c(r)
           rule / pre-projection / ceiling) x seeds 0-2 x 12000 steps, shadow on,
           svd_driver auto; plus a same-trajectory fixed-rank sweep
           --fixed_r in {4,8,16,24,32,48,64} (n=64; 48/64 need
           --force_preproject 1 --c 8 to avoid the r_max=n Corollary-3 collapse
           that silently drops the sketch machinery, per algos.py SKRTRL.__init__).
  d3_oat   rotation, henon, sunspot x 10 one-at-a-time variants (default + each
           of tau_high, tau_low, M, K perturbed one at a time, plus the
           rank-interval [2,64] control with --c 8 --force_preproject 1 to hold
           c fixed) x seeds 0-2 x 12000 steps, shadow on, svd_driver auto. TS
           tasks (henon, sunspot) carry horizon/causal/washout like run_m3.py.

Usage
-----
  python make_r2_d3_jobs.py --stage diag --out jobs/d3_diag.txt
  python make_r2_d3_jobs.py --stage oat --out jobs/d3_oat.txt
  python make_r2_d3_jobs.py --stage all
  python make_r2_d3_jobs.py --status
"""
import argparse
import os

from make_r2_jobs import BATCH, LOG_EVERY, TS_TASKS, REAL_TASKS, WASHOUT, HORIZON

PY_ADAPT = "run_adaptive.py"

DIAG_TASKS = ["rotation", "anbn"]
OAT_TASKS = ["rotation", "henon", "sunspot"]
SEEDS_D3 = [0, 1, 2]
STEPS_D3 = 12000

N_DEFAULT = 64
DEFAULT_LR = 1e-3          # run_adaptive.py's own --lr default; not ablated in D3
MS_PER_STEP = 30.0         # run_adaptive ~ same order of magnitude as skrtrl-r16 (sec. 2)

# controller defaults (must match run_adaptive.py argparse defaults)
R_MIN_DEFAULT = 4
R_MAX_DEFAULT = 32
K_DEFAULT = 100
M_DEFAULT = 3
TAU_LOW_DEFAULT = 0.003
TAU_HIGH_DEFAULT = 0.02
CLIP_DEFAULT = 0.5
CTRL_DEFAULT = "eta"

FIXED_R_SWEEP = [4, 8, 16, 24, 32, 48, 64]


def adapt_out_path(outdir, task, seed, tag, fixed_r=None, ctrl=CTRL_DEFAULT):
    base = f"fixed{fixed_r}" if (fixed_r is not None and fixed_r > 0) else f"adaptive-{ctrl}"
    tag_suffix = f"_{tag}" if tag else ""
    return f"{outdir}/{task}_{base}_s{seed}{tag_suffix}.json"


def adapt_job(task, seed, steps, outdir, tag, *, lr=DEFAULT_LR, n=N_DEFAULT,
             r_min=R_MIN_DEFAULT, r_max=R_MAX_DEFAULT, K=K_DEFAULT, M=M_DEFAULT,
             tau_low=TAU_LOW_DEFAULT, tau_high=TAU_HIGH_DEFAULT, clip=CLIP_DEFAULT,
             ctrl=CTRL_DEFAULT, c=None, force_preproject=None, preproject=None,
             fixed_r=None, svd_driver="auto", shadow=1, extra=""):
    parts = [PY_ADAPT,
             f"--task {task}", f"--steps {steps}", f"--n {n}", f"--batch {BATCH}",
             f"--lr {lr:g}", f"--seed {seed}", f"--log_every {LOG_EVERY}",
             f"--shadow {shadow}", f"--svd_driver {svd_driver}"]
    if fixed_r is not None:
        parts.append(f"--fixed_r {fixed_r}")
    parts += [f"--r_min {r_min}", f"--r_max {r_max}", f"--K {K}", f"--M {M}",
              f"--tau_low {tau_low:g}", f"--tau_high {tau_high:g}", f"--clip {clip:g}",
              f"--ctrl {ctrl}"]
    if c is not None:
        parts.append(f"--c {c}")
    if force_preproject is not None:
        parts.append(f"--force_preproject {int(bool(force_preproject))}")
    if preproject is not None:
        parts.append(f"--preproject {preproject}")
    if task in TS_TASKS:
        causal = 1 if task in REAL_TASKS else 0
        parts += [f"--horizon {HORIZON}", f"--causal {causal}", f"--washout {WASHOUT}"]
    parts += [f"--outdir {outdir}", f"--tag {tag}"]
    if extra:
        parts.append(extra)
    out = adapt_out_path(outdir, task, seed, tag, fixed_r=fixed_r, ctrl=ctrl)
    return " ".join(parts) + f"  # out={out}"


# --- D3 diag: variants A-E (one-at-a-time around the c(r) rule / pre-projection /
# ceiling), rotation + anbn only. kwargs are passed straight to adapt_job. ---
DIAG_VARIANTS = [
    ("vA_default", dict()),
    ("vB_rmax64_c8fp", dict(r_max=64, force_preproject=1, c=8)),
    ("vC_n128_rmax64", dict(n=128, r_max=64)),
    ("vD_preprojoff", dict(preproject="off")),
    ("vE_c8_rmax32", dict(c=8)),
]


def stage_diag():
    outdir = "results/r2/d3_diag"
    n_variant_jobs = len(DIAG_TASKS) * len(DIAG_VARIANTS) * len(SEEDS_D3)
    n_sweep_jobs = len(DIAG_TASKS) * len(FIXED_R_SWEEP) * len(SEEDS_D3)
    n_jobs = n_variant_jobs + n_sweep_jobs
    gpu_h = n_jobs * STEPS_D3 * MS_PER_STEP / 1000.0 / 3600.0
    jobs = [f"# D3 diag stage: {len(DIAG_TASKS)} tasks {DIAG_TASKS} x "
            f"{len(DIAG_VARIANTS)} kernel variants (A-E) x {len(SEEDS_D3)} seeds x "
            f"{STEPS_D3} steps -> {n_variant_jobs} jobs, plus same-trajectory "
            f"fixed_r sweep {FIXED_R_SWEEP} x {len(DIAG_TASKS)} tasks x "
            f"{len(SEEDS_D3)} seeds -> {n_sweep_jobs} jobs (total {n_jobs})",
            f"# estimated GPU-h at {MS_PER_STEP:.0f} ms/step (run_adaptive ~ same "
            f"order as skrtrl-r16 shadow-on auto, sec. 2): {gpu_h:.1f} "
            f"(variant C runs at n=128 and is likely somewhat slower than this "
            f"uniform estimate)",
            "# variants: A default c(r) rule r_max=32; B r_max=64 --force_preproject "
            "1 --c 8 (n=64); C n=128 r_max=64 (default c rule); D pre-projection off "
            "(--preproject off) r_max=32; E fixed c=8 r_max=32",
            "# fixed_r sweep: same task/seed trajectory, --fixed_r in "
            f"{FIXED_R_SWEEP} (n=64); r_max=32 for fixed_r<=32, r_max=64 + "
            "--force_preproject 1 --c 8 for fixed_r in {48, 64} (avoids the "
            "r_max=n Corollary-3 collapse)"]
    for task in DIAG_TASKS:
        for tag, kw in DIAG_VARIANTS:
            for seed in SEEDS_D3:
                jobs.append(adapt_job(task, seed, STEPS_D3, outdir, tag, **kw))
    for task in DIAG_TASKS:
        for fr in FIXED_R_SWEEP:
            r_max = 64 if fr in (48, 64) else 32
            kw = dict(fixed_r=fr, r_max=r_max)
            if fr in (48, 64):
                kw.update(force_preproject=1, c=8)
            for seed in SEEDS_D3:
                jobs.append(adapt_job(task, seed, STEPS_D3, outdir, "sweep", **kw))
    return jobs


# --- D3 oat: one-at-a-time sensitivity around the defaults, rotation/henon/sunspot. ---
OAT_VARIANTS = [
    ("default", dict()),
    ("tauhigh0.01", dict(tau_high=0.01)),
    ("tauhigh0.04", dict(tau_high=0.04)),
    ("taulow0.0015", dict(tau_low=0.0015)),
    ("taulow0.006", dict(tau_low=0.006)),
    ("M2", dict(M=2)),
    ("M5", dict(M=5)),
    ("K50", dict(K=50)),
    ("K200", dict(K=200)),
    ("interval2-64_c8fp", dict(r_min=2, r_max=64, c=8, force_preproject=1)),
]


def stage_oat():
    outdir = "results/r2/d3_oat"
    n_jobs = len(OAT_TASKS) * len(OAT_VARIANTS) * len(SEEDS_D3)
    gpu_h = n_jobs * STEPS_D3 * MS_PER_STEP / 1000.0 / 3600.0
    jobs = [f"# D3 oat stage: {len(OAT_TASKS)} tasks {OAT_TASKS} x "
            f"{len(OAT_VARIANTS)} one-at-a-time variants x {len(SEEDS_D3)} seeds x "
            f"{STEPS_D3} steps, shadow on, svd_driver auto -> {n_jobs} jobs",
            f"# estimated GPU-h at {MS_PER_STEP:.0f} ms/step (measured order, sec. 2): "
            f"{gpu_h:.1f}",
            "# variants: default; tau_high in {0.01, 0.04}; tau_low in "
            "{0.0015, 0.006}; M in {2, 5}; K in {50, 200}; rank interval [2, 64] "
            "with --c 8 --force_preproject 1 to keep c fixed"]
    for task in OAT_TASKS:
        for tag, kw in OAT_VARIANTS:
            for seed in SEEDS_D3:
                jobs.append(adapt_job(task, seed, STEPS_D3, outdir, tag, **kw))
    return jobs


STAGES = {"diag": stage_diag, "oat": stage_oat}
DEFAULT_OUT = {"diag": "jobs/d3_diag.txt", "oat": "jobs/d3_oat.txt"}


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


def status():
    for name, fn in STAGES.items():
        lines = fn()
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
    ap.add_argument("--stage", choices=["diag", "oat", "all"], default="all")
    ap.add_argument("--out", default=None, help="output path (default per stage)")
    ap.add_argument("--limit", type=int, default=0,
                    help="keep only the first N real jobs per stage (smoke tests)")
    ap.add_argument("--status", action="store_true",
                    help="print expected/present job counts and exit (no write)")
    args = ap.parse_args()

    if args.status:
        status()
        return

    stages = ["diag", "oat"] if args.stage == "all" else [args.stage]
    for name in stages:
        lines = STAGES[name]()
        out_path = args.out if (args.out and len(stages) == 1) else DEFAULT_OUT[name]
        write_jobs(lines, out_path, args.limit)


if __name__ == "__main__":
    main()
