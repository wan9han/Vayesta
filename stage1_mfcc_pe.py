#!/usr/bin/env python3
"""Stage 1 — first physical MFCC vs full-system energy comparison on PE.

For a minimal polyethylene chain (ethane C2H6 and butane C4H10) we compute:

    E_full   = SIESTA energy of the whole molecule
    E_MFCC   = Σ E(capped fragment) − Σ E(conjugate cap)

and report the error ``E_MFCC − E_full``. This is the first physical test of
the project goal ("fragmented energy reproduces full-system energy") and the
first number that cannot be hidden behind a manifest flag.

All SIESTA runs use identical numerical settings (SZ / LDA-PZ / MeshCutoff
100 Ry / explicit 8 Å vacuum cell), so any gap is a genuine fragmentation
(cap) error, not a numerical bias.

Run:
    python stage1_mfcc_pe.py \
        --siesta-bin /home/xzz2/huawei-siesta/siesta-install/bin/siesta \
        --pseudo-dir /home/xzz2/huawei-siesta/testcases \
        --gen-script /home/xzz2/huawei-siesta/testcases/gen.py \
        --work-root  /tmp/stage1-runs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from energy_first.mfcc_geometry import build_mfcc_cut, pick_middle_cut
from energy_first.molecule import Molecule, parse_gen_fdf_text
from energy_first.siesta_backend import run_siesta


def _fingerprint(mol: Molecule) -> str:
    arr = np.round(np.asarray(mol.coords), 8)
    payload = "".join(mol.elements) + "|" + arr.tobytes().hex()
    return hashlib.sha256(payload.encode()).hexdigest()


def gen_molecule(python: str, gen_script: str, n_carbons: int) -> Molecule:
    proc = subprocess.run(
        [python, gen_script, str(n_carbons)],
        capture_output=True, text=True, check=True,
    )
    return parse_gen_fdf_text(proc.stdout, label=f"PE_{n_carbons}C_full")


def run_with_cache(mol, cache, run_dir_root, **run_kwargs):
    """Run SIESTA once per distinct geometry; reuse cached energy."""
    fp = _fingerprint(mol)
    if fp in cache:
        return cache[fp]
    run_dir = run_dir_root / f"run_{len(cache):04d}_{mol.label}"
    res = run_siesta(mol, run_dir, **run_kwargs)
    res["fingerprint"] = fp
    cache[fp] = res
    return res


def run_case(n_carbons, ctx) -> dict:
    run_dir_root = ctx["work_root"] / f"pe_{n_carbons}C"
    run_dir_root.mkdir(parents=True, exist_ok=True)

    full = gen_molecule(ctx["python"], ctx["gen_script"], n_carbons)
    full.label = f"PE_{n_carbons}C_full"

    full_res = run_with_cache(
        full, ctx["cache"], run_dir_root,
        siesta_bin=ctx["siesta_bin"], pseudo_dir=ctx["pseudo_dir"],
        label=full.label,
    )
    e_full = full_res["energy_ev"]

    cut = pick_middle_cut(full)
    fragments, caps = build_mfcc_cut(full, cut)

    frag_res = [
        run_with_cache(f, ctx["cache"], run_dir_root,
                       siesta_bin=ctx["siesta_bin"], pseudo_dir=ctx["pseudo_dir"],
                       label=f"PE_{n_carbons}C_frag{idx}")
        for idx, f in enumerate(fragments)
    ]
    cap_res = [
        run_with_cache(c, ctx["cache"], run_dir_root,
                       siesta_bin=ctx["siesta_bin"], pseudo_dir=ctx["pseudo_dir"],
                       label=f"PE_{n_carbons}C_cap{idx}")
        for idx, c in enumerate(caps)
    ]

    missing = None
    if any(r["energy_ev"] is None for r in [full_res] + frag_res + cap_res):
        missing = [
            r["label"] for r in [full_res] + frag_res + cap_res if r["energy_ev"] is None
        ]
        e_mfcc = None
        error_ev = None
    else:
        e_mfcc = sum(r["energy_ev"] for r in frag_res) - sum(r["energy_ev"] for r in cap_res)
        error_ev = e_mfcc - e_full

    natoms = full.natoms
    return {
        "n_carbons": n_carbons,
        "formula": full_res["formula"],
        "natoms": natoms,
        "cut_pair": list(cut),
        "E_full_ev": e_full,
        "fragments": [{"label": r["label"], "formula": r["formula"], "E_ev": r["energy_ev"]}
                       for r in frag_res],
        "caps": [{"label": r["label"], "formula": r["formula"], "E_ev": r["energy_ev"]}
                 for r in cap_res],
        "E_mfcc_ev": e_mfcc,
        "error_ev": error_ev,
        "error_per_atom_ev": (error_ev / natoms) if error_ev is not None else None,
        "failed_runs": missing,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--siesta-bin", default="/home/xzz2/huawei-siesta/siesta-install/bin/siesta")
    ap.add_argument("--pseudo-dir", default="/home/xzz2/huawei-siesta/testcases")
    ap.add_argument("--gen-script", default="/home/xzz2/huawei-siesta/testcases/gen.py")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--work-root", default="/tmp/stage1-runs")
    ap.add_argument("--cases", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)

    ctx = {
        "siesta_bin": args.siesta_bin,
        "pseudo_dir": args.pseudo_dir,
        "gen_script": args.gen_script,
        "python": args.python,
        "work_root": Path(args.work_root),
        "cache": {},
    }

    cases = [run_case(n, ctx) for n in args.cases]
    report = {
        "stage": "stage1_mfcc_pe",
        "method": "MFCC single middle cut, H caps, conjugate cap H2",
        "siesta": args.siesta_bin,
        "settings": "SZ / LDA-PZ / MeshCutoff 100 Ry / explicit 8A vacuum cell / Diagonali",
        "formula": "E_MFCC = sum(E_capped_fragment) - sum(E_conjugate_cap)",
        "cases": cases,
    }
    out = Path(args.report) if args.report else ctx["work_root"] / "stage1_mfcc_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"\n=== Stage 1 MFCC vs full (report -> {out}) ===")
    for c in cases:
        print(f"\n-- {c['formula']} (n_carbons={c['n_carbons']}, natoms={c['natoms']}), cut={c['cut_pair']}")
        print(f"   E_full   = {c['E_full_ev']}")
        for fr in c["fragments"]:
            print(f"   frag {fr['label']} ({fr['formula']}) = {fr['E_ev']}")
        for cp in c["caps"]:
            print(f"   cap  {cp['label']} ({cp['formula']}) = {cp['E_ev']}")
        print(f"   E_MFCC   = {c['E_mfcc_ev']}")
        print(f"   error    = {c['error_ev']} eV  ({c['error_per_atom_ev']:.4f} eV/atom)"
              if c["error_per_atom_ev"] is not None else "   error    = N/A")
        if c["failed_runs"]:
            print(f"   FAILED: {c['failed_runs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
