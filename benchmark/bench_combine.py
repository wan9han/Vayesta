#!/usr/bin/env python3
"""收集(combine)性能 benchmark：用 16×500 真实结果做模板，复制到 N 节点规模，
执行 MFCC(1)+MBE(2) 收集计算并计时。规模: 1/2/4/8/16/22680。"""
import json, time, re, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
tmpl = json.loads((ROOT / "template_16x500.json").read_text())
TB, TC, TD = tmpl["blocks"], tmpl["caps"], tmpl["dimers"]
TOTAL_RE = re.compile(r"siesta:.*Total\s*=\s*(-?\d+\.\d+)")
SCALES = [1, 2, 4, 8, 16, 22680]

def fabricate(n, out):
    """造 N 节点的 siesta.out（cycle 模板能量）+ schedule.json。"""
    if out.exists():
        return  # 复用（不重复造）
    out.mkdir(parents=True)
    nc = n - 1
    sched = {"num_nodes": n, "dimers": [{"dimer_id": i} for i in range(nc)]}
    (out / "schedule.json").write_text(json.dumps(sched))
    for i in range(n):
        d = out / f"block_{i:04d}"; d.mkdir(exist_ok=True)
        (d / "siesta.out").write_text(f"siesta:         Total =     {TB[i % len(TB)]:.6f}\n")
    for i in range(nc):
        for kind, pool in (("cap", TC), ("dimer", TD)):
            d = out / f"{kind}_{i:04d}"; d.mkdir(exist_ok=True)
            (d / "siesta.out").write_text(f"siesta:         Total =     {pool[i % len(pool)]:.6f}\n")

def combine(out):
    """复刻 combine_results.py 的收集逻辑，返回 (e_mbe2, 耗时)。"""
    root = out
    sched = json.loads((root / "schedule.json").read_text())
    n = sched["num_nodes"]; nc = n - 1; nd = len(sched.get("dimers", []))
    t0 = time.perf_counter()
    be = []
    for i in range(n):
        m = TOTAL_RE.findall((root / f"block_{i:04d}" / "siesta.out").read_text())
        be.append(float(m[-1]))
    ce = [float(TOTAL_RE.findall((root / f"cap_{i:04d}" / "siesta.out").read_text())[-1]) for i in range(nc)]
    de = [float(TOTAL_RE.findall((root / f"dimer_{i:04d}" / "siesta.out").read_text())[-1]) for i in range(nd)] if nd == nc else []
    e_mfcc = sum(be) - sum(ce)
    e_mbe2 = e_mfcc
    if nd == nc:
        for k in range(nd):
            e_mbe2 += de[k] - be[k] - be[k + 1] + ce[k]
    return e_mbe2, time.perf_counter() - t0

print(f"{'nodes':>6} | {'fabricate':>10} | {'combine(s)':>10} | {'files':>7} | E_total(eV)")
print("-" * 65)
for n in SCALES:
    out = ROOT / f"combine_{n}"
    t0 = time.perf_counter(); fabricate(n, out); tf = time.perf_counter() - t0
    e, tc = combine(out)
    nf = n + 2 * (n - 1)
    print(f"{n:>6} | {tf:>9.2f}s | {tc:>10.4f} | {nf:>7} | {e:,.1f}")
