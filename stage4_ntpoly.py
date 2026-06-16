#!/usr/bin/env python3
"""Stage 4 — partitioning works with SIESTA's production solver (ELSI/NTPoly).

Two checks:

(A) Solver consistency: for the same molecule, Diagonali and NTPoly (TRS2)
    must give the same SCF-converged energy (different density-matrix solver,
    same physics). This confirms NTPoly is a drop-in for the backend.

(B) Energy reproduction with NTPoly: run the MFCC + MBE(2) partitioning with
    the NTPoly backend and confirm E^(2) still matches E_full — i.e. the
    correct energy recombination is compatible with the project's large-block
    sparse solver, not only with Diagonali.

This bridges correctness (partitioned == full) and the project's efficiency
engine (NTPoly, the path used for large blocks).
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
    ap.add_argument("--work-root", default="/tmp/stage4-runs")
    ap.add_argument("--basis", default="SZ")
    ap.add_argument("--solver-chains", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--repro-chain", type=int, default=8)
    ap.add_argument("--repro-frags", type=int, nargs="+", default=[2, 3, 4])
    args = ap.parse_args(argv)

    root = Path(args.work_root)
    root.mkdir(parents=True, exist_ok=True)
    cache: dict = {}

    def run(mol, label, method):
        mol = mol.__class__(list(mol.elements), mol.coords.copy(), label)
        key = (args.basis, method, _fp(mol))
        if key in cache:
            return cache[key]
        d = root / f"{method}_{len(cache):04d}_{label}"
        r = run_siesta(mol, d, siesta_bin=args.siesta_bin, pseudo_dir=args.pseudo_dir,
                       label=label, basis_size=args.basis, solution_method=method)
        cache[key] = r["energy_ev"]
        return r["energy_ev"]

    print(f"\n=== Stage 4: NTPoly backend, basis={args.basis} ===")
    print("--- (A) solver consistency: Diagonali vs NTPoly (same molecule) ---")
    print(f"{'molecule':>10} {'E_diagonali':>16} {'E_ntpoly':>16} {'diff(eV)':>12}")
    for n in args.solver_chains:
        mol = gen(args.python, args.gen_script, n)
        ed = run(mol, f"C{n}_full", "diagonali")
        en = run(mol, f"C{n}_full", "ntpoly")
        if ed is None or en is None:
            print(f"{'C%dH%d' % (n, 2*n+2):>10}  FAILED (diagonali=%s ntpoly=%s)" % (ed, en))
            continue
        print(f"{'C%dH%d' % (n, 2*n+2):>10} {ed:>16.4f} {en:>16.4f} {en - ed:>12.5f}")

    print("\n--- (B) MFCC + MBE(2) reproduction with NTPoly backend ---")
    mol = gen(args.python, args.gen_script, args.repro_chain)
    e_full_nt = run(mol, f"C{args.repro_chain}_full", "ntpoly")
    print(f"{'E_full(ntpoly)':>24} = {e_full_nt:.4f} eV")
    print(f"{'case':>10} {'cuts':>5} {'E^(1)':>14} {'err1':>11} {'E^(2)':>14} {'err2':>11}")
    for nf in args.repro_frags:
        cuts = pick_cuts(mol, nf)

        def run_cached(m, _method="ntpoly"):
            return run(m, f"C{args.repro_chain}_f{nf}", _method)

        res = mbe2_over_capped_fragments(mol, cuts, run_cached)
        e1, e2 = res["E_mbe1_ev"], res["E_mbe2_ev"]
        print(f"{'C%d:%df' % (args.repro_chain, nf):>10} {len(cuts):>5} {e1:>14.4f} {e1 - e_full_nt:>11.4f}"
              f" {e2:>14.4f} {e2 - e_full_nt:>11.4f}")
    print(f"\n(distinct SIESTA jobs this run: {len(cache)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
