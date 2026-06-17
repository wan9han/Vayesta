#!/usr/bin/env python3
"""Stage 5 — 386-atom polyethylene (C128H258) partitioning verification.

The project's standard large system. Genuine split + energy reproduction:
runs E_full (NTPoly, the benchmark) then an N-fragment MBE(2) whose dimers
are ~half the full system -> genuine matrix split. Reports, per sub-task, the
matrix size NTPoly actually received (proving the big matrix is split), and
the E_MBE(2) residual vs E_full.

Run (from the algorithm repo root, alongside the energy_first package):
    PYTHONPATH=. python stage5_386.py \\
        --siesta-bin /path/to/siesta \\
        --pseudo-dir /path/to/psf \\
        --gen-script /path/to/gen.py \\
        --python /path/to/python
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from energy_first.mbe_mfcc import mbe2_over_capped_fragments
from energy_first.mfcc_geometry import pick_cuts
from energy_first.molecule import Molecule, parse_gen_fdf_text
from energy_first.siesta_backend import run_siesta


def _fp(mol):
    arr = np.round(np.asarray(mol.coords), 8)
    return hashlib.sha256(("".join(mol.elements) + "|" + arr.tobytes().hex()).encode()).hexdigest()


def _matrix_size(run_dir):
    """Parse SIESTA 'Number of atoms, orbitals, and projectors' line.

    Columns (0-indexed after split): [7]=natoms, [8]=norbitals (= matrix
    dimension NTPoly solves), [9]=nprojectors.
    """
    out = os.path.join(run_dir, "siesta.out")
    if not os.path.exists(out):
        return None
    with open(out) as fh:
        for line in fh:
            if "Number of atoms, orbitals, and projectors" in line:
                f = line.split()
                return {"natoms": int(f[7]), "norbitals": int(f[8]), "nprojectors": int(f[9])}
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--siesta-bin", default="/home/xzz2/huawei-siesta/siesta-install/bin/siesta")
    ap.add_argument("--pseudo-dir", default="/home/xzz2/huawei-siesta/testcases")
    ap.add_argument("--gen-script", default="/home/xzz2/huawei-siesta/testcases/gen.py")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--work-root", default="/tmp/stage5-386")
    ap.add_argument("--carbons", type=int, default=128, help="gen.py arg; 128 -> C128H258 (386 atoms)")
    ap.add_argument("--fragments", type=int, default=4)
    ap.add_argument("--basis", default="SZ")
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    root = Path(args.work_root)
    root.mkdir(parents=True, exist_ok=True)
    cache = {}

    def run(mol, label, timeout=3600):
        mol = Molecule(list(mol.elements), mol.coords.copy(), label)
        key = (args.basis, "ntpoly", _fp(mol))
        if key in cache:
            return cache[key]
        d = root / f"ntpoly_{len(cache):04d}_{label}"
        r = run_siesta(mol, d, args.siesta_bin, args.pseudo_dir, label=label,
                       basis_size=args.basis, solution_method="ntpoly", timeout=timeout)
        cache[key] = (r["energy_ev"], str(d))
        ms = _matrix_size(str(d))
        no = ms["norbitals"] if ms else "?"
        print(f"  [{label}] natoms={r['natoms']} norbitals={no} E={r['energy_ev']} eV", flush=True)
        return r["energy_ev"], str(d)

    print(f"=== Stage 5: C{args.carbons}H{2*args.carbons+2} PE, "
          f"{args.fragments} fragments, NTPoly, basis={args.basis} ===", flush=True)
    p = subprocess.run([args.python, args.gen_script, str(args.carbons)],
                       capture_output=True, text=True, check=True)
    mol = parse_gen_fdf_text(p.stdout, label=f"C{args.carbons}_full")
    print(f"generated C{args.carbons}H{2*args.carbons+2}: {mol.natoms} atoms", flush=True)

    print("[1/2] E_full (NTPoly) — the benchmark...", flush=True)
    t0 = time.time()
    e_full, full_dir = run(mol, f"C{args.carbons}_full", timeout=3600)
    m_full = _matrix_size(full_dir)
    print(f"  E_full = {e_full} eV | full matrix = {m_full} | {time.time()-t0:.0f}s", flush=True)

    NF = args.fragments
    print(f"[2/2] MBE(2) with {NF} fragments (genuine split)...", flush=True)
    cuts = pick_cuts(mol, NF)
    t0 = time.time()
    res = mbe2_over_capped_fragments(
        mol, cuts, lambda m: run(m, f"C{args.carbons}_f{NF}", timeout=3000)[0]
    )
    e1, e2 = res["E_mbe1_ev"], res["E_mbe2_ev"]

    # sub-molecule matrix sizes NTPoly actually received (proves the split)
    sizes = []
    for (e, d) in cache.values():
        ms = _matrix_size(d)
        if ms:
            sizes.append({"natoms": ms["natoms"], "norbitals": ms["norbitals"]})
    sizes.sort(key=lambda s: s["norbitals"], reverse=True)

    print(f"  E_MBE1 = {e1}  err1 = {e1 - e_full} eV", flush=True)
    print(f"  E_MBE2 = {e2}  err2 = {e2 - e_full} eV | {time.time()-t0:.0f}s", flush=True)
    print(f"  full matrix norbitals = {m_full['norbitals'] if m_full else '?'}", flush=True)
    print(f"  largest sub-matrix norbitals = {sizes[0]['norbitals'] if sizes else '?'} "
          f"({sizes[0]['norbitals']/m_full['norbitals']:.1%} of full)" if sizes and m_full else "", flush=True)

    report = {
        "system": f"C{args.carbons}H{2*args.carbons+2}", "solver": "NTPoly(TRS2)", "basis": args.basis,
        "E_full_ev": e_full, "full_matrix": m_full,
        "num_fragments": NF, "num_cuts": len(cuts),
        "E_mbe1_ev": e1, "err1_ev": e1 - e_full,
        "E_mbe2_ev": e2, "err2_ev": e2 - e_full,
        "sub_matrix_sizes_sorted": sizes,
    }
    out = Path(args.report) if args.report else root / "stage5_386_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("DONE -> " + str(out), flush=True)


if __name__ == "__main__":
    main()
