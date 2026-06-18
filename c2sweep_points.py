import glob, json, math, sys


def tail(recs, k, frac=0.3):
    v = [r[k] for r in recs if r.get(k) is not None and not (isinstance(r[k], float) and math.isnan(r[k]))]
    return sum(v[-max(1, int(len(v) * frac)):]) / max(1, int(len(v) * frac)) if v else None


d_ = sys.argv[1] if len(sys.argv) > 1 else "results/c2sweep"
rows = []
for f in sorted(glob.glob(f"{d_}/*.json")):
    d = json.load(open(f))
    a = d["args"]
    rb = tail(d["records"], "rho_bar")
    rh = tail(d["records"], "rho_hat")
    e = tail(d["records"], "e_t")
    E = tail(d["records"], "true_E")
    if rb and e and E and E > 0:
        rows.append((a["task"], a.get("clip"), rb, rh, e / E))
rows.sort(key=lambda x: (x[0], x[2]))
print("task           clip  rho_bar  rho_hat   tightness")
for t, c, rb, rh, tg in rows:
    print(f"{t:14s} {c:5.2f}  {rb:7.3f}  {rh:7.3f}  {tg:10.2f}")
