# Acceptance run on a fresh clone (v1.0)

Date: 2026-10-15. Commit cloned: `d8b403fb49d989197a0e251f5c396dd5c2187c0e` (the release commit before this log was added; this
file and the README/REPRODUCIBILITY references to it are the only later change).

Machine: Windows 10 Pro 10.0.19045 workstation, two NVIDIA GeForce RTX 3080 (20 GB), driver
560.94; the GPU step used GPU0 (`CUDA_VISIBLE_DEVICES=0`). Environment: Python 3.10.20, PyTorch
2.6.0+cu124 (CUDA 12.4, cuDNN 9.1.0), NumPy 1.26.4, SciPy 1.15.3, Matplotlib 3.10.9, pytest 9.1.1;
commands ran in Git Bash. This is the workstation of the second campaign (REPRODUCIBILITY.md,
section 1), so the re-run below is on the same card model, driver and library versions as the
released record.

The clone was made into an empty directory with nothing next to it, so `../paper/secs` is absent
and the manuscript comparison of `check_release.py` is skipped (and counted as skipped). In our
working copy, where the manuscript sources sit next to the repository, the same commit gives
`check_release: 147 ok, 0 skipped, 0 failed`.

## Commands, return codes and wall times

| # | Command (from the clone root) | Return code | Wall time (s) |
|---|---|---|---|
| 1 | `git clone <repository> accept && cd accept` | 0 | 15.3 |
| 2 | `python -m pip install --dry-run -r requirements.txt` | 0 | 5.6 |
| 3 | `python -m pytest tests -q --import-mode=importlib` | 0 | 53.0 |
| 4 | `python check_release.py` | 0 | 79.4 |
| 5 | `python make_paper_figures.py` | 0 | 19.9 |
| 6 | `CUDA_VISIBLE_DEVICES=0 python run_m3.py --task adding --algo skrtrl-r16 --seed 0 --lr 0.003 --steps 20000 --n 64 --batch 8 --log_every 250 --shadow 1 --svd_driver auto --outdir results/rerun/d4_eval --tag lr3e-03` | 0 | 960.8 |
| 7 | `python compare_rerun.py   # listed below` | 0 | 0.5 |
| 8 | `python dry_run_manifest.py jobs/d4_eval.txt   # listed below` | 0 | 0.4 |
| 9 | `grep -v '^#' jobs/d4_eval.txt \| head -3 \| sed -e 's/  # out=.*//' \| while read -r cmd; do python $cmd; done` | 0 | 13.7 |

Installation: the environment already satisfied every line of `requirements.txt`, so the dry run
(`pip install --dry-run`) was used; it reported every requirement as already satisfied and would
install nothing.

## Results

- `git status` right after the clone: (clean).
- pytest: `16 passed, 4 warnings in 50.90s`.
- check_release.py: `check_release: 138 ok, 1 skipped, 0 failed -- nothing failed; the SKIP lines above were not checked`. Skipped: `SKIP ../paper/secs not found next to the repository: table fragments, Table 7 rows and Table S10 NOT compared with the manuscript`.
  Failed: none.
- make_paper_figures.py: exit code as above; no figure skipped.

### GPU re-run of the worked example against its released record

The run is the first SK-RTRL r=16 evaluation run of `jobs/d4_eval.txt` (adding, seed 0, tuned
rate 3e-3, 20,000 steps, exact shadow on), written to `results/rerun/d4_eval/`. The comparison
reads the mean over the records in the last 20% of the run, as every table does, from the re-run
and from `results/r2/d4_eval/adding_skrtrl-r16_s0_lr3e-03.json`, and puts the difference beside
the spread of the ten released seeds of the same pair.

```
metric    released 0.206200  rerun 0.206200  diff +0.00e+00  | released seeds 0-9: mean 0.211826 sd 0.011093 min 0.196575 max 0.225986 (n=10)
grad_cos  released 0.888955  rerun 0.888955  diff +0.00e+00  | released seeds 0-9: mean 0.852511 sd 0.041830 min 0.789160 max 0.928840 (n=10)
cert_summary.n_steps            released 198  rerun 198
cert_summary.n_bound_finite     released 0  rerun 0
cert_summary.n_bound_violation  released 0  rerun 0
cert_summary.n_cert_invalid     released 0  rerun 0
cert_summary.frac_rel_lt1       released 0.0  rerun 0.0
args identical apart from outdir: True
first logged record released: {'step': 500, 'metric': 0.9904621958732605, 'grad_cos': 0.8668842196464539, 'rho_bar': 4.71199893951416, 'rho_hat': 0.7496448755264282, 'eta': 0.0, 'e_t': 0.0, 'true_E': 0.0, 'J_norm': 9.056207656860352, 'delta_norm': 0.1185562014579773, 'ghat_norm': 0.21016830205917358, 'bound_norm': inf, 'g_norm': 0.27546828985214233, 'gerr_norm': 0.14114639163017273, 'diag_step': 499}
first logged record rerun:    {'step': 500, 'metric': 0.9904621958732605, 'grad_cos': 0.8668842196464539, 'rho_bar': 4.71199893951416, 'rho_hat': 0.7496448755264282, 'eta': 0.0, 'e_t': 0.0, 'true_E': 0.0, 'J_norm': 9.056207656860352, 'delta_norm': 0.1185562014579773, 'ghat_norm': 0.21016830205917358, 'bound_norm': inf, 'g_norm': 0.27546828985214233, 'gerr_norm': 0.14114639163017273, 'diag_step': 499}
wall_s released 323 rerun 951; peak_MB released 52.6435546875 rerun 52.6435546875
```

Observed: both final-window numbers agree with the released record to every printed digit
(difference 0), and so do all 79 logged records, the certificate counters and the SVD-driver
counts (40,000 default-driver calls, no fallback). The run took 951 s against 323 s for the
released run. This re-run used the card model, driver and library versions of the second
campaign; bit-identity is what we observed here, not what the release promises. On other hardware
or library versions, expect agreement within the seed-to-seed spread shown above (for this pair,
s.d. 0.011 in the metric and 0.042 in gradient cosine over seeds 0-9).

Last lines of the run:

```
[adding/skrtrl-r16/s0] step 18000 metric 0.2062 cos 0.8886538439326817
[adding/skrtrl-r16/s0] sketch@19999 r16 eff 1.9516397373457508 resfrac 13.8
saved results/rerun/d4_eval/adding_skrtrl-r16_s0_lr3e-03.json (951s, 53MB)
```

### Manifest dry run (the exists-skip path)

`dry_run_manifest.py` parses every command line of `jobs/d4_eval.txt` and checks that the file
named by its `# out=` annotation exists, which is the condition under which the runners and
`run_r2_queue.sh` skip a job:

```
jobs/d4_eval.txt: 810 command lines, scripts ['run_m3.py']; 810 outputs exist, 0 to run
```

The first three lines of the manifest, run as written (output into the released tree), each
exit 0 without training:

```
exists, skip: results/r2/d4_eval/adding_exact_s0_lr3e-03.json
exists, skip: results/r2/d4_eval/adding_snap1_s0_lr1e-02.json
exists, skip: results/r2/d4_eval/adding_skrtrl-r4_s0_lr3e-03.json
```

`git status` after all steps: (clean). (`results/rerun/` and the regenerated figures are ignored
by `.gitignore`.)

## Helper scripts used above

`compare_rerun.py`:

```python
"""Compare the GPU re-run of the worked example with its released record."""
import glob, json, statistics
REL = "results/r2/d4_eval/adding_skrtrl-r16_s0_lr3e-03.json"
NEW = "results/rerun/d4_eval/adding_skrtrl-r16_s0_lr3e-03.json"


def window(path, field):
    a = json.load(open(path, encoding="utf-8"))
    steps = a["args"]["steps"]
    v = [r[field] for r in a["records"] if r.get("step", -1) >= 0.8 * steps
         and isinstance(r.get(field), (int, float))]
    return sum(v) / len(v), a


for field in ("metric", "grad_cos"):
    rel, a = window(REL, field)
    new, b = window(NEW, field)
    seeds = [window(f, field)[0] for f in sorted(glob.glob(
        "results/r2/d4_eval/adding_skrtrl-r16_s*_lr3e-03.json"))]
    print("%-9s released %.6f  rerun %.6f  diff %+.2e  | released seeds 0-9: mean %.6f sd %.6f "
          "min %.6f max %.6f (n=%d)" % (field, rel, new, new - rel, statistics.mean(seeds),
                                        statistics.stdev(seeds), min(seeds), max(seeds), len(seeds)))
ca, cb = a["cert_summary"], b["cert_summary"]
for k in ("n_steps", "n_bound_finite", "n_bound_violation", "n_cert_invalid", "frac_rel_lt1"):
    print("cert_summary.%-18s released %s  rerun %s" % (k, ca.get(k), cb.get(k)))
print("args identical apart from outdir:",
      {k: v for k, v in a["args"].items() if k != "outdir"} ==
      {k: v for k, v in b["args"].items() if k != "outdir"})
print("first logged record released:", a["records"][1])
print("first logged record rerun:   ", b["records"][1])
print("wall_s released %.0f rerun %.0f; peak_MB released %s rerun %s" % (
    a.get("wall_s", 0), b.get("wall_s", 0), a.get("peak_MB"), b.get("peak_MB")))
```

`dry_run_manifest.py`:

```python
"""Dry run of a manifest: parse every line, confirm its '# out=' file exists (exists-skip path)."""
import os, sys
path = sys.argv[1]
n = have = 0
missing = []
scripts = set()
for line in open(path, encoding="utf-8"):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    cmd, _, out = line.partition("# out=")
    toks = cmd.split()
    scripts.add(toks[1] if toks[0] == "python" else toks[0])
    n += 1
    if os.path.isfile(out.strip()):
        have += 1
    else:
        missing.append(out.strip())
print("%s: %d command lines, scripts %s; %d outputs exist, %d to run" % (
    path, n, sorted(scripts), have, len(missing)))
for m in missing[:5]:
    print("missing:", m)
sys.exit(1 if missing else 0)
```

The clone was deleted after the run.
