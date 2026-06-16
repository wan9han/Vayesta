#!/usr/bin/env python3
"""Stage 1c — does a better basis reduce the per-cut MFCC error?

Runs the single-cut MFCC test on ethane/butane at SZ vs DZP (everything else
identical: LDA-PZ, MeshCutoff, explicit vacuum cell) to see whether the ~1.2
eV/cut cap error is basis-limited or inherent to the H-cap scheme.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np

from energy_first.mfcc_geometry import build_mfcc_cut, pick_middle_cut
from energy_first.molecule import parse_gen_fdf_text
from energy_first.siesta_backend import run_siesta


def _fp(el, coords):
    arr = np.round(np.asarray(coords), 8)
    return hashlib.sha256((el + "|" + arr.tobytes().hex()).encode()).hexdigest()


def gen(python, gen_script, n):
    p = subprocess.run([python, gen_script, str(n)], capture_output=True, text=True, check=True)
    return parse_gen_fdf_text(p.stdout, label=f"PE_{n}C")


def run(mol, cache, root, ctx, label, basis):
    mol2 = mol.__class__(list(mol.elements), mol.coords.copy(), label)
    key = (basis, _fp("".join(mol2.elements), mol2.coords))
    if key in cache:
        return cache[key]
    d = root / f"{basis}_{len(cache):04d}_{label}"
    r = run_siesta(mol2, d, siesta_bin=ctx["bin"], pseudo_dir=ctx["psd"],
                   label=label, basis_size=basis)
    cache[key] = r["energy_ev"]
    return r["energy_ev"]


def case(n, basis, ctx):
    root = ctx["root"] / f"c{n}_{basis}"
    root.mkdir(parents=True, exist_ok=True)
    full = gen(ctx["py"], ctx["gen"], n)
    e_full = run(full, ctx["cache"], root, ctx, f"C{n}_full", basis)
    cut = pick_middle_cut(full)
    frags, caps = build_mfcc_cut(full, cut)
    efs = [run(f, ctx["cache"], root, ctx, f"C{n}_frag{i}_{basis}", basis) for i, f in enumerate(frags)]
    ecs = [run(c, ctx["cache"], root, ctx, f"C{n}_cap{i}_{basis}", basis) for i, c in enumerate(caps)]
    if any(e is None for e in [e_full] + efs + ecs):
        return None
    e_mfcc = sum(efs) - sum(ecs)
    return {"n": n, "basis": basis, "E_full": e_full, "E_mfcc": e_mfcc,
            "err": e_mfcc - e_full, "per_atom": (e_mfcc - e_full) / full.natoms}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--siesta-bin", default="/home/xzz2/huawei-siesta/siesta-install/bin/siesta")
    ap.add_argument("--pseudo-dir", default="/home/xzz2/huawei-siesta/testcases")
    ap.add_argument("--gen-script", default="/home/xzz2/huawei-siesta/testcases/gen.py")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--work-root", default="/tmp/stage1c-runs")
    ap.add_argument("--chains", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--bases", nargs="+", default=["SZ", "DZP"])
    args = ap.parse_args(argv)

    ctx = {"bin": args.siesta_bin, "psd": args.pseudo_dir, "gen": args.gen_script,
           "py": args.python, "root": Path(args.work_root), "cache": {}}

    print(f"\n=== Stage 1c: per-cut MFCC error vs basis ===")
    print(f"{'chain':>8} {'basis':>6} {'E_full':>14} {'E_MFCC':>14} {'err(eV)':>12} {'eV/atom':>10}")
    for n in args.chains:
        for b in args.bases:
            r = case(n, b, ctx)
            if r is None:
                print(f"{'C%d'%(n):>8} {b:>6}  FAILED")
                continue
            print(f"{'C%dH%d'%(n,2*n+2):>8} {b:>6} {r['E_full']:>14.4f} {r['E_mfcc']:>14.4f} "
                  f"{r['err']:>12.4f} {r['per_atom']:>10.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
