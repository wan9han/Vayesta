#!/usr/bin/env python3
"""Generation (展开) benchmark: time weak_scale_pe.py generation at given node counts.

Portable -- no hardcoded paths. gen-script / pseudo-dir are required args;
pythonpath is optional (set only if your python3 lacks numpy/scipy and you
must point to a venv). The subprocess inherits the caller's environment.

Examples:
  # intranet (system python3 already has numpy/scipy):
  python3 bench_gen.py --gen-script /path/to/gen.py --pseudo-dir /path/to/pseudos
  # dev box (numpy lives in a venv):
  python3 bench_gen.py --gen-script /path/gen.py --pseudo-dir /path/pseudos \\
      --pythonpath /path/to/venv/site-packages
  # full scale (slow, ~28 min):
  python3 bench_gen.py --gen-script ... --pseudo-dir ... --nodes 22680
"""
import argparse
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VAYESTA = ROOT.parent  # weak_scale_pe.py lives one level up


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--gen-script", required=True,
                    help="path to the PE chain generator (gen.py)")
    ap.add_argument("--pseudo-dir", required=True,
                    help="dir containing C.psf and H.psf")
    ap.add_argument("--pythonpath", default=None,
                    help="PYTHONPATH for numpy/scipy (omit if system python3 has them)")
    ap.add_argument("--nodes", type=int, nargs="+", default=[1, 2, 4, 8, 16],
                    help="node counts to benchmark (default: 1 2 4 8 16)")
    ap.add_argument("--atoms-per-node", type=int, default=8000)
    args = ap.parse_args()

    env = dict(os.environ)
    if args.pythonpath:
        env["PYTHONPATH"] = args.pythonpath + (
            ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )

    print(f"{'nodes':>6} | {'wall':>9} | [gen-time]")
    print("-" * 90)
    for n in args.nodes:
        out = ROOT / f"gen_{n}"
        if out.exists():
            shutil.rmtree(out)
        hosts = [f"n{i}" for i in range(n)]
        cmd = [
            "python3", str(VAYESTA / "weak_scale_pe.py"),
            "--atoms-per-node", str(args.atoms_per_node),
            "--num-nodes", str(n),
            "--pseudo-dir", args.pseudo_dir,
            "--out-dir", str(out),
            "--gen-script", args.gen_script,
            "--no-full-baseline",
            "--hosts", *hosts,
        ]
        t0 = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(VAYESTA))
        wall = time.perf_counter() - t0
        m = re.search(r"\[gen-time\] (.*)", r.stdout)
        detail = m.group(1) if m else (r.stdout[-200:] + r.stderr[-200:])
        print(f"{n:>6} | {wall:>8.2f}s | {detail}")


if __name__ == "__main__":
    main()
