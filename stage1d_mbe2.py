#!/usr/bin/env python3
"""Stage 1d — MBE(2) over capped fragments: how much does it cut the error?

For ethane/butane/octane we compare:
  E_full : whole-molecule SIESTA energy (benchmark)
  E^(1)  : plain MFCC (1-body), ~1.2 eV/cut
  E^(2)  : 2-body MBE over capped segments (joined dimers + inclusion-exclusion)

Sanity: at 2 fragments, E^(2) == E_full exactly (N=2 identity). At 3/4
fragments the residual is the 3-body term, expected to be far below 1 eV/cut.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np

from energy_first.mbe_mfcc import mbe2_over_capped_fragments
from energy_first.mfcc_geometry import pick_cuts
from energy_first.molecule import parse_gen_fdf_text
from energy_first.siesta_backend import run_siesta


def _fp(mol):
    arr = np.round(np.asarray(mol.coords), 8)
    return hashlib.sha256(("".join(mol.elements) + "|" + arr.tobytes().hex()).encode()).hexdigest()


def gen(python, gen_script, n):
    p = subprocess.run([python, gen_script, str(n)], capture_output=True, text=True, check=True)
    return parse_gen_fdf_text(p.stdout, label=f"PE_{n}C_full")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--siesta-bin", default="/home/xzz2/huawei-siesta/siesta-install/bin/siesta")
    ap.add_argument("--pseudo-dir", default="/home/xzz2/huawei-siesta/testcases")
    ap.add_argument("--gen-script", default="/home/xzz2/huawei-siesta/testcases/gen.py")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--work-root", default="/tmp/stage1d-runs")
    ap.add_argument("--basis", default="SZ")
    ap.add_argument("--solution-method", default="diagonali",
                    choices=["diagonali", "ntpoly"])
    ap.add_argument("--cases", type=str, default="2:2,4:2,8:2,8:3,8:4",
                    help="comma list of n_carbons:num_fragments")
    args = ap.parse_args(argv)

    root = Path(args.work_root)
    root.mkdir(parents=True, exist_ok=True)
    cache: dict = {}

    def run(mol, label):
        mol = mol.__class__(list(mol.elements), mol.coords.copy(), label)
        key = (args.basis, args.solution_method, _fp(mol))
        if key in cache:
            return cache[key]
        d = root / f"{args.solution_method}_{len(cache):04d}_{label}"
        r = run_siesta(mol, d, siesta_bin=args.siesta_bin, pseudo_dir=args.pseudo_dir,
                       label=label, basis_size=args.basis, solution_method=args.solution_method)
        cache[key] = r["energy_ev"]
        return r["energy_ev"]

    cases = []
    for tok in args.cases.split(","):
        n, nf = tok.split(":")
        cases.append((int(n), int(nf)))

    print(f"\n=== Stage 1d: MBE(1) vs MBE(2) vs E_full, basis={args.basis} ===")
    print(f"{'case':>10} {'cuts':>5} {'E_full':>14} {'E^(1)':>14} {'err1(eV)':>11}"
          f" {'E^(2)':>14} {'err2(eV)':>11} {'improve':>9}")
    for n, nf in cases:
        mol = gen(args.python, args.gen_script, n)
        e_full = run(mol, f"C{n}_full")
        cuts = pick_cuts(mol, nf)
        label = f"C{n}_f{nf}"

        def run_cached(m, _label=label):
            return run(m, _label)

        res = mbe2_over_capped_fragments(mol, cuts, run_cached)
        e1, e2 = res["E_mbe1_ev"], res["E_mbe2_ev"]
        err1 = e1 - e_full
        err2 = e2 - e_full
        improve = (abs(err1) / abs(err2)) if err2 not in (0, None) else float("inf")
        print(f"{'C%d:%df' % (n, nf):>10} {len(cuts):>5} {e_full:>14.4f} {e1:>14.4f} {err1:>11.4f}"
              f" {e2:>14.4f} {err2:>11.4f} {improve:>8.1f}x")
    print(f"\n(distinct SIESTA jobs this run: {len(cache)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
