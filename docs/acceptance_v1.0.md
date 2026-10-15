# Acceptance run on a fresh clone (v1.0)

This file records two acceptance runs. The first (below) was made at commit
`d8b403fb49d989197a0e251f5c396dd5c2187c0e`, the parent of the tagged release commit
`583dc9fdd1c26bed4ad7571852cf8420340613c2` (`v1.0-neurocomputing`); the tag commit differs from it
only by this file (`git diff --stat d8b403f 583dc9f`: `docs/acceptance_v1.0.md | 171 +++`, 1 file
changed, 171 insertions). The second, "Checked out at the tag", was made on a fresh clone checked
out at the tag itself. The helper scripts of the second run are kept as `tools/compare_rerun.py` and
`tools/dry_run_manifest.py`; they, and the second section, were added on `main` after the tag,
without changing code, data or the release check.

## First run: commit d8b403f

Date: 2026-10-15. Commit cloned: `d8b403fb49d989197a0e251f5c396dd5c2187c0e`.

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

### Commands, return codes and wall times

| # | Command (from the clone root) | Return code | Wall time (s) |
|---|---|---|---|
| 1 | `git clone <repository> accept && cd accept` | 0 | 15.3 |
| 2 | `python -m pip install --dry-run -r requirements.txt` | 0 | 5.6 |
| 3 | `python -m pytest tests -q --import-mode=importlib` | 0 | 53.0 |
| 4 | `python check_release.py` | 0 | 79.4 |
| 5 | `python make_paper_figures.py` | 0 | 19.9 |
| 6 | `CUDA_VISIBLE_DEVICES=0 python run_m3.py --task adding --algo skrtrl-r16 --seed 0 --lr 0.003 --steps 20000 --n 64 --batch 8 --log_every 250 --shadow 1 --svd_driver auto --outdir results/rerun/d4_eval --tag lr3e-03` | 0 | 960.8 |
| 7 | `python compare_rerun.py   # the earlier, shorter version of tools/compare_rerun.py; see below` | 0 | 0.5 |
| 8 | `python dry_run_manifest.py jobs/d4_eval.txt   # identical to tools/dry_run_manifest.py` | 0 | 0.4 |
| 9 | `grep -v '^#' jobs/d4_eval.txt \| head -3 \| sed -e 's/  # out=.*//' \| while read -r cmd; do python $cmd; done` | 0 | 13.7 |

Installation: the environment already satisfied every line of `requirements.txt`, so the dry run
(`pip install --dry-run`) was used; it reported every requirement as already satisfied and would
install nothing.

### Results

- `git status` right after the clone: (clean).
- pytest: `16 passed, 4 warnings in 50.90s`.
- check_release.py: `check_release: 138 ok, 1 skipped, 0 failed -- nothing failed; the SKIP lines above were not checked`. Skipped: `SKIP ../paper/secs not found next to the repository: table fragments, Table 7 rows and Table S10 NOT compared with the manuscript`.
  Failed: none.
- make_paper_figures.py: exit code as above; no figure skipped.

#### GPU re-run of the worked example against its released record

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

What the comparison script of this run checked, and observed: the two final-window means (equal
to every printed digit, difference 0); the five certificate counters printed above (equal); the
arguments apart from `outdir` (equal); one logged record, the one at step 500 (equal in every
field); and the peak memory (equal). It did not compare the other 78 logged records, the other
certificate fields or the SVD-driver counts, and the re-run JSON of this run was not kept. The
record-by-record comparison, with `tools/compare_rerun.py`, is in the second section below. The
run took 951 s against 323 s for the released run. This re-run used the card model, driver and library versions of the second
campaign; bit-identity is what we observed here, not what the release promises. On other hardware
or library versions, expect agreement within the seed-to-seed spread shown above (for this pair,
s.d. 0.011 in the metric and 0.042 in gradient cosine over seeds 0-9).

Last lines of the run:

```
[adding/skrtrl-r16/s0] step 18000 metric 0.2062 cos 0.8886538439326817
[adding/skrtrl-r16/s0] sketch@19999 r16 eff 1.9516397373457508 resfrac 13.8
saved results/rerun/d4_eval/adding_skrtrl-r16_s0_lr3e-03.json (951s, 53MB)
```

#### Manifest dry run (the exists-skip path)

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

The comparison script of this run (step 7) printed only the lines shown above. It has been
replaced by `tools/compare_rerun.py`, which compares everything it compared and every other
logged value as well (second section); `tools/dry_run_manifest.py` is the script of step 8 unchanged.

The clone was deleted after the run.

## Second run: checked out at the tag

Date: 2026-10-15 (14:21-14:40 UTC+8). Same machine, GPU, driver, Python environment and shell as the
first run. A fresh clone of the public repository (`https://github.com/sonaradarcn/sketched-rtrl.git`)
was made into an empty directory and checked out at the tag; as in the first run, `../paper/secs`
is absent, so the manuscript comparison of `check_release.py` is skipped. The tag does not contain
`tools/`, so steps 10 and 11 ran `tools/compare_rerun.py` and `tools/dry_run_manifest.py` as
committed on `main` together with this section, from the clone root.

### Commands, return codes and wall times

| # | Command (from the clone root) | Return code | Wall time (s) |
|---|---|---|---|
| 1 | `git clone https://github.com/sonaradarcn/sketched-rtrl.git sketched-rtrl && cd sketched-rtrl` | 0 | 35.0 |
| 2 | `git fetch --tags` | 0 | 2.5 |
| 3 | `git checkout v1.0-neurocomputing` | 0 | 0.3 |
| 4 | `git rev-parse HEAD` | 0 | 0.2 |
| 5 | `git describe --tags --exact-match HEAD` | 0 | 0.2 |
| 6 | `python -m pip install --dry-run -r requirements.txt` | 0 | 2.0 |
| 7 | `python check_release.py` | 0 | 80.9 |
| 8 | `python -m pytest tests -q --import-mode=importlib` | 0 | 52.3 |
| 9 | `CUDA_VISIBLE_DEVICES=0 python run_m3.py --task adding --algo skrtrl-r16 --seed 0 --lr 0.003 --steps 20000 --n 64 --batch 8 --log_every 250 --shadow 1 --svd_driver auto --outdir results/rerun/d4_eval --tag lr3e-03` | 0 | 936.1 |
| 10 | `python tools/compare_rerun.py` | 0 | 0.7 |
| 11 | `python tools/dry_run_manifest.py jobs/d4_eval.txt` | 0 | 0.4 |

### Results

- `git rev-parse HEAD`: `583dc9fdd1c26bed4ad7571852cf8420340613c2`; `git describe --tags --exact-match HEAD`: `v1.0-neurocomputing`.
- `git status` after the checkout: (clean). After all steps: (clean).
- pip dry run: every requirement already satisfied; nothing would be installed.
- check_release.py: `check_release: 138 ok, 1 skipped, 0 failed -- nothing failed; the SKIP lines above were not checked`. Skipped: `SKIP ../paper/secs not found next to the repository: table fragments, Table 7 rows and Table S10 NOT compared with the manuscript`. Failed: none.
- pytest: `16 passed, 4 warnings in 50.31s`.
- GPU re-run: `saved results/rerun/d4_eval/adding_skrtrl-r16_s0_lr3e-03.json (929s, 53MB)`.

#### Record-by-record comparison of the re-run with its released record

`tools/compare_rerun.py` compares the re-run JSON with
`results/r2/d4_eval/adding_skrtrl-r16_s0_lr3e-03.json` value by value: the number of logged
records and every field of every record; every field of `cert_summary` (the certificate counters);
`svd_stats` and `svd_fallbacks` (the SVD-driver counts); every field of each sketch checkpoint; and
every argument apart from `outdir`. A numeric pair is exact when the two floats are equal (or both
NaN); otherwise it is within tolerance when both are finite and `|a - b| <= 1e-12 + 1e-6 |a|`, and
different if not. A record is identical when all its fields are exact. `wall_s` and `peak_MB` are
printed and not compared. Output:

```
released: results/r2/d4_eval/adding_skrtrl-r16_s0_lr3e-03.json
rerun:    results/rerun/d4_eval/adding_skrtrl-r16_s0_lr3e-03.json
tolerance: |a - b| <= 1e-12 + 1e-06 * |a| (finite values); exact counted separately
records       79 released, 79 rerun; 79 records compared (1179 values): 79 identical, 0 within tolerance only, 0 different
cert_summary  28 fields: identical
svd_stats     5 fields: identical
svd_fallbacks 3 fields: identical
svd_stats released {'default': 40000, 'gesvd': 0, 'gesvd_after_default': 0, 'jitter': 0, 'cpu': 0}
svd_stats rerun    {'default': 40000, 'gesvd': 0, 'gesvd_after_default': 0, 'jitter': 0, 'cpu': 0}
checkpoints   6 compared: 6 identical, 0 within tolerance only, 0 different
args          29 fields apart from outdir: identical
metric    released 0.206200  rerun 0.206200  diff +0.00e+00  | released seeds: mean 0.211826 sd 0.011093 min 0.196575 max 0.225986 (n=10)
grad_cos  released 0.888955  rerun 0.888955  diff +0.00e+00  | released seeds: mean 0.852511 sd 0.041830 min 0.789160 max 0.928840 (n=10)
wall_s released 323 rerun 929; peak_MB released 52.6435546875 rerun 52.6435546875 (not compared)
all values compared: 1364 exact, 0 within tolerance only, 0 different
compare_rerun: PASS
```

Observed: all 79 logged records are identical to the released record in every field (1,179
values), as are the 28 certificate fields, the SVD-driver counts (40,000 default-driver calls, no
fallback), the 6 checkpoints and the 29 arguments other than `outdir`: 1,364 values exact, none
merely within tolerance, none different. The run took 929 s against 323 s for the released run. As
for the first run, bit-identity is what we observed on the hardware and library versions of the
second campaign, not what the release promises.

#### Manifest dry run

```
jobs/d4_eval.txt: 810 command lines, scripts ['run_m3.py']; 810 outputs exist, 0 to run
```

This checks only that every output named in the manifest exists, which is the condition under which
the runners skip a job; it retrains nothing.

The clone was deleted after the run.
