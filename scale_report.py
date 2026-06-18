import glob, json, math, sys
from collections import defaultdict
d_ = sys.argv[1] if len(sys.argv) > 1 else "results/scale"
def tail(recs,k,frac=0.3):
    v=[r[k] for r in recs if r.get(k) is not None and not(isinstance(r[k],float) and math.isnan(r[k]))]
    return sum(v[-max(1,int(len(v)*frac)):])/max(1,int(len(v)*frac)) if v else None
runs=defaultdict(list)
for f in sorted(glob.glob(f"{d_}/*.json")):
    d=json.load(open(f)); a=d["args"]
    runs[(a["n"],a["task"],a["algo"])].append({"cos":tail(d["records"],"grad_cos"),"MB":d.get("peak_MB")})
def avg(rs,x):
    v=[r[x] for r in rs if r[x] is not None]; return sum(v)/len(v) if v else None
print("n    task      algo         grad_cos   peak_MB")
for k in sorted(runs):
    rs=runs[k]; c=avg(rs,"cos"); m=avg(rs,"MB")
    cs=f"{c:.3f}" if c is not None else "  -  "
    ms=f"{m:.0f}" if m is not None else " - "
    print(f"{k[0]:<4} {k[1]:<9} {k[2]:<12} {cs:>8}   {ms:>7}")
