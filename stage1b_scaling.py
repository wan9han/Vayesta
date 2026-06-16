#!/usr/bin/env python3
"""Stage 1b — MFCC error scaling (the weak-scaling-relevant evidence).

Two scans, both honest SIESTA numbers (SZ/LDA, identical numerics):

  (A) Single middle cut vs chain length (C2, C4, C6, C8, ...):
      error should be ~constant — it is a *per-cut* error, independent of
      chain length. Per-atom error therefore falls as 1/natoms.

  (B) Octane C8H18 vs number of fragments (2, 3, 4, ...):
      total error should grow ~linearly with the number of cuts
      (num_fragments - 1), because each cut contributes ~a constant cap error.

Together: error ≈ (num_cuts) × (per-cut error). With LARGE fragments the
number of cuts is small relative to system size, so the fractional/per-atom
error stays bounded and small as the system grows — the regime the
weak-scaling goal needs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from energy_first.mfcc_geometry import build_mfcc, pick_cuts
from energy_first.molecule import Molecule, parse_gen_fdf_text
from energy_first.siesta_backend import run_siesta


def _fingerprint(mol: Molecule) -> str:
    arr = np.round(np.asarray(mol.coords), 8)
    payload = "".join(mol.elements) + "|" + arr.tobytes().hex()
    return hashlib.sha256(payload.encode()).hexdigest()


def gen_molecule(python, gen_script, n_carbons):
    proc = subprocess.run([python, gen_script, str(n_carbons)],
                          capture_output=True, text=True, check=True)
    return parse_gen_fdf_text(proc.stdout, label=f"PE_{n_carbons}C_full")


def _run(mol, cache, run_root, ctx, label):
    fp = _fingerprint(mol)
    if fp in cache:
        return cache[fp]
    d = run_root / f"run_{len(cache):04d}_{label}"
    res = run_siesta(mol, d, siesta_bin=ctx["siesta_bin"],
                     pseudo_dir=ctx["pseudo_dir"], label=label)
    cache[fp] = res
    return res


def compare(n_carbons, num_fragments, ctx):
    run_root = ctx["work_root"] / f"scan_C{n_carbons}_f{num_fragments}"
    run_root.mkdir(parents=True, exist_ok=True)
    full = gen_molecule(ctx["python"], ctx["gen_script"], n_carbons)
    full.label = f"C{n_carbons}_full"

    e_full = _run(full, ctx["cache"], run_root, ctx, full.label)["energy_ev"]
    cuts = pick_cuts(full, num_fragments)
    frags, caps = build_mfcc(full, cuts)

    fr_energies = [
        _run(f, ctx["cache"], run_root, ctx, f"C{n_carbons}_f{num_fragments}_frag{i}")["energy_ev"]
        for i, f in enumerate(frags)
    ]
    cap_energies = [
        _run(c, ctx["cache"], run_root, ctx, f"C{n_carbons}_f{num_fragments}_cap{i}")["energy_ev"]
        for i, c in enumerate(caps)
    ]

    ok = all(e is not None for e in [e_full] + fr_energies + cap_energies)
    e_mfcc = (sum(fr_energies) - sum(cap_energies)) if ok else None
    err = (e_mfcc - e_full) if ok else None
    return {
        "n_carbons": n_carbons, "natoms": full.natoms,
        "num_fragments": num_fragments, "num_cuts": len(cuts),
        "cuts": [list(c) for c in cuts],
        "E_full_ev": e_full, "E_mfcc_ev": e_mfcc,
        "error_ev": err,
        "error_per_atom_ev": (err / full.natoms) if err is not None else None,
        "ok": ok,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--siesta-bin", default="/home/xzz2/huawei-siesta/siesta-install/bin/siesta")
    ap.add_argument("--pseudo-dir", default="/home/xzz2/huawei-siesta/testcases")
    ap.add_argument("--gen-script", default="/home/xzz2/huawei-siesta/testcases/gen.py")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--work-root", default="/tmp/stage1b-runs")
    ap.add_argument("--single-cut-chains", type=int, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--multifrag-carbons", type=int, default=8)
    ap.add_argument("--multifrag-frags", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)

    ctx = {"siesta_bin": args.siesta_bin, "pseudo_dir": args.pseudo_dir,
           "gen_script": args.gen_script, "python": args.python,
           "work_root": Path(args.work_root), "cache": {}}

    print(f"\n=== (A) single middle cut vs chain length ===")
    print(f"{'chain':>10} {'natoms':>7} {'E_full':>14} {'E_MFCC':>14} {'err(eV)':>12} {'eV/atom':>10}")
    scan_a = []
    for n in args.single_cut_chains:
        r = compare(n, 2, ctx)
        scan_a.append(r)
        ea = f"{r['error_per_atom_ev']:.5f}" if r["error_per_atom_ev"] is not None else "n/a"
        print(f"{'C%dH%d' % (n, 2*n+2):>10} {r['natoms']:>7} {r['E_full_ev']:>14.4f} "
              f"{r['E_mfcc_ev']:>14.4f} {r['error_ev']:>12.4f} {ea:>10}")

    print(f"\n=== (B) C{args.multifrag_carbons} vs number of fragments (cuts) ===")
    print(f"{'frags':>6} {'cuts':>6} {'E_full':>14} {'E_MFCC':>14} {'err(eV)':>12} {'eV/atom':>10}")
    scan_b = []
    for nf in args.multifrag_frags:
        r = compare(args.multifrag_carbons, nf, ctx)
        scan_b.append(r)
        ea = f"{r['error_per_atom_ev']:.5f}" if r["error_per_atom_ev"] is not None else "n/a"
        print(f"{nf:>6} {r['num_cuts']:>6} {r['E_full_ev']:>14.4f} {r['E_mfcc_ev']:>14.4f} "
              f"{r['error_ev']:>12.4f} {ea:>10}")

    report = {
        "stage": "stage1b_scaling",
        "scan_A_single_cut_vs_chain": scan_a,
        "scan_B_multifragment_C8": scan_b,
        "note": "error ~ per-cut; grows linearly with num cuts, amortized over atoms",
    }
    out = Path(args.report) if args.report else ctx["work_root"] / "stage1b_scaling_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nreport -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
