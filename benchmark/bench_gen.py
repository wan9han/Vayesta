#!/usr/bin/env python3
"""展开(生成)性能 benchmark：在 1/2/4/8/16 节点跑 weak_scale_pe 生成，计时。
22680 节点因耗时(~28min)建议单独后台跑：python3 bench_gen.py 22680"""
import subprocess, sys, time, re, os, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parent
VAYESTA = ROOT.parent
GEN = "/home/xzz2/huawei-siesta/testcases/gen.py"
PSEUDO = "/home/xzz2/huawei-siesta/testcases"
PP = "/tmp/detopt_venv/lib/python3.10/site-packages"
SCALES = [int(x) for x in sys.argv[1:]] or [1, 2, 4, 8, 16]

def run_gen(n):
    out = ROOT / f"gen_{n}"
    if out.exists(): shutil.rmtree(out)
    env = {**os.environ, "PYTHONPATH": PP}
    hosts = [f"n{i}" for i in range(n)]
    cmd = ["python3", str(VAYESTA/"weak_scale_pe.py"), "--atoms-per-node","8000",
           "--num-nodes",str(n), "--pseudo-dir",PSEUDO, "--out-dir",str(out),
           "--gen-script",GEN, "--no-full-baseline", "--hosts",*hosts]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(VAYESTA))
    wall = time.perf_counter() - t0
    m = re.search(r"\[gen-time\] (.*)", r.stdout)
    return wall, (m.group(1) if m else r.stdout[-300:]+r.stderr[-300:])

print(f"{'nodes':>6} | {'wall':>8} | [gen-time]")
print("-"*80)
for n in SCALES:
    wall, gt = run_gen(n)
    print(f"{n:>6} | {wall:>7.2f}s | {gt}")
