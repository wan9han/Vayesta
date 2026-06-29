#!/usr/bin/env python3
"""Combine (收集) benchmark -- real filesystem impact, portable.

Replicates REAL siesta.out templates (block ~1.4MB / dimer ~2.5MB / cap ~27KB,
captured from an actual intranet run under templates/) to N nodes, drops in the
real combine_results.py, and runs it. This measures the true cost of collecting
N nodes' outputs: opening/reading ~68000 real-size files + regex + sum, i.e. the
filesystem load at full scale -- NOT a toy with 1-line stubs.

No hardcoded paths. Usage:
  python3 bench_combine.py                   # 1/2/4/8/16 (seconds)
  python3 bench_combine.py --nodes 22680     # full scale (~90GB, ~10min)
"""
import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TPL = ROOT / "templates"


def fabricate(n, out):
    """Copy real templates into N-node layout + schedule.json. Returns (files, bytes)."""
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TPL / "combine_results.py", out / "combine_results.py")
    nc = n - 1
    (out / "schedule.json").write_text(
        json.dumps({"num_nodes": n, "dimers": [{"dimer_id": i} for i in range(nc)]})
    )
    bsz = (TPL / "block.out").stat().st_size
    dsz = (TPL / "dimer.out").stat().st_size
    csz = (TPL / "cap.out").stat().st_size
    nfiles = nbytes = 0
    for i in range(n):
        d = out / f"block_{i:04d}"
        d.mkdir(exist_ok=True)
        shutil.copyfile(TPL / "block.out", d / "siesta.out")
        nfiles += 1; nbytes += bsz
    for i in range(nc):
        d = out / f"cap_{i:04d}"
        d.mkdir(exist_ok=True)
        shutil.copyfile(TPL / "cap.out", d / "siesta.out")
        nfiles += 1; nbytes += csz
        d = out / f"dimer_{i:04d}"
        d.mkdir(exist_ok=True)
        shutil.copyfile(TPL / "dimer.out", d / "siesta.out")
        nfiles += 1; nbytes += dsz
    return nfiles, nbytes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nodes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    args = ap.parse_args()

    print(f"{'nodes':>6} | {'files':>7} | {'bytes':>9} | {'fabricate':>9} | {'combine':>9}")
    print("-" * 62)
    for n in args.nodes:
        out = ROOT / f"combine_{n}"
        if out.exists():
            shutil.rmtree(out)
        t0 = time.perf_counter()
        nf, nb = fabricate(n, out)
        tf = time.perf_counter() - t0
        t1 = time.perf_counter()
        r = subprocess.run(["python3", "combine_results.py"],
                           capture_output=True, text=True, cwd=str(out))
        tc = time.perf_counter() - t1
        ct = re.search(r"\[combine-time\] (.*)", r.stdout)
        print(f"{n:>6} | {nf:>7} | {nb / 1e9:>8.2f}G | {tf:>8.1f}s | {tc:>8.2f}s"
              + (f"   [{ct.group(1).strip()}]" if ct else ""))


if __name__ == "__main__":
    main()
