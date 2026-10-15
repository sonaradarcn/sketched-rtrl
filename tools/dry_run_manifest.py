"""Dry run of a manifest: parse every line, confirm its '# out=' file exists (exists-skip path).

Run from the repository root:  python tools/dry_run_manifest.py jobs/d4_eval.txt
Exit status 0 when every output named by a '# out=' annotation exists, 1 otherwise.
"""
import os
import sys

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
