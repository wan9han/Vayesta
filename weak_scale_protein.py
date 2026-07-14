#!/usr/bin/env python3
"""Weak-scaling polyglycine generator (protein branch).

Mirrors weak_scale_pe.py but for polyglycine (C/H/N/O, C'-N peptide bond
cuts). Reuses weak_scale_pe's general infrastructure (FDF writing, launch
scripts, combine template, scheduling) and protein_validate's fragmentation
(peptide bond detection, union-find, H caps, dimers).

Usage:
  python3 weak_scale_protein.py --residues-per-node 50 --num-nodes 4 --out-dir /tmp/ws_gly4
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from weak_scale_pe import (
    DEFAULT_HOSTS,
    _write,
    _slot_ranges,
    _place_pseudo,
    _write_inputs,
    _write_launch_artifacts,
)
from energy_first.peptide_chain import generate_polyglycine
from energy_first.molecule import Molecule
from protein_validate import (
    find_peptide_bonds,
    fragment,
    build_dimer,
)


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--residues-per-node", type=int, required=True,
                    help="glycine residues per node (~7 atoms/residue)")
    ap.add_argument("--num-nodes", type=int, required=True)
    ap.add_argument("--procs-per-node", type=int, default=16)
    ap.add_argument("--gen-script", default=str(Path(__file__).resolve().parent / "gen.py"),
                    help="(unused for protein; kept for compat)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument(
        "--pseudo-dir",
        default=str(Path(__file__).resolve().parent / "pseudos"),
        help="dir with C/H/N/O .psf (default: repo pseudos/)",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--basis", default="SZ")
    ap.add_argument("--mesh-cutoff-ry", type=float, default=100.0)
    ap.add_argument("--block-slice-num", type=int, default=1,
                    help="NTPOLY_SLICE_NUM for blocks (default 1)")
    ap.add_argument("--dimer-slice-num", type=int, default=2,
                    help="NTPOLY_SLICE_NUM for dimers (default 2)")
    ap.add_argument(
        "--shared-pseudo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Share pseudos via symlink (default on).",
    )
    ap.add_argument(
        "--full-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate unfragmented full-chain baseline (default on).",
    )
    ap.add_argument(
        "--solution-method",
        default="ntpoly",
        choices=["ntpoly", "diagonali"],
    )
    ap.add_argument("--hosts", nargs="+", default=DEFAULT_HOSTS)
    ap.add_argument("--num-numa", type=int, default=16)
    ap.add_argument("--cores-per-numa", type=int, default=38)
    ap.add_argument("--skip-cores", type=int, default=2)
    ap.add_argument("--omp-threads", type=int, default=None)
    ap.add_argument("--pkg-root", default="/share/honpas/xzz/siesta-20260520")
    ap.add_argument("--mpi-prefix", default="/share/hmpi2.4.1/hmpi-v2.4.1-huawei")
    ap.add_argument("--siesta-app", default=None)
    ap.add_argument("--env-sh", default=None)
    ap.add_argument("--launch-orted", default=None)
    ap.add_argument("--remote-out-dir", default=None)
    ap.add_argument("--ssh-user", default="")
    return ap.parse_args()


def _generate_chain(args):
    """Generate polyglycine chain. Returns (mol, n_residues)."""
    total = args.residues_per_node * args.num_nodes
    elements, coords = generate_polyglycine(total)
    mol = Molecule(list(elements), coords, label=f"gly{total}")
    return mol, total


def _build_fragments(mol, n_residues, args):
    """Fragment polyglycine at peptide bonds. Returns (schedule, block_mols, cap_mols, dimer_mols)."""
    bonds = find_peptide_bonds(mol)
    num_nodes = args.num_nodes
    if num_nodes == 1:
        cuts = []
    else:
        # Evenly-spaced peptide bonds (same approach as PE's pick_cuts)
        n_bonds = len(bonds)
        cuts = []
        for k in range(1, num_nodes):
            idx = min(round(k * n_bonds / num_nodes), n_bonds - 1)
            cuts.append(bonds[idx])

    frag_mols, cap_mols, comp = fragment(mol, cuts)
    dimer_mols = []
    if getattr(args, "mbe_order", 2) >= 2 and cuts:
        for k, cut in enumerate(cuts):
            d = build_dimer(mol, cuts, cut)
            d.label = f"dimer_{k:04d}"
            dimer_mols.append(d)

    # Relabel fragments as block_XXXX (matching weak_scale_pe convention)
    for i, f in enumerate(frag_mols):
        f.label = f"block_{i:04d}"
    for i, c in enumerate(cap_mols):
        c.label = f"cap_{i:04d}"

    schedule = {
        "residues_per_node": args.residues_per_node,
        "num_nodes": num_nodes,
        "procs_per_node": args.procs_per_node,
        "num_numa": args.num_numa,
        "omp_threads": args.omp_threads,
        "total_chain_atoms": mol.natoms,
        "total_residues": n_residues,
        "hosts": args.hosts[:num_nodes],
        "cuts": [list(c) for c in cuts],
        "blocks": [],
        "caps": [],
        "dimers": [],
        "mbe_order": 2,
        "system": "polyglycine",
    }
    for i, f in enumerate(frag_mols):
        schedule["blocks"].append({
            "block_id": i,
            "host": args.hosts[i],
            "natoms": f.natoms,
        })
    for i, c in enumerate(cap_mols):
        schedule["caps"].append({
            "cap_id": i,
            "natoms": c.natoms,
            "host": args.hosts[0],
        })
    for k, d in enumerate(dimer_mols):
        schedule["dimers"].append({
            "dimer_id": k,
            "cut": list(cuts[k]),
            "left_block": k,
            "right_block": k + 1,
            "host": args.hosts[k % num_nodes],
            "natoms": d.natoms,
        })

    return schedule, frag_mols, cap_mols, dimer_mols


def main():
    args = _parse_args()
    if args.num_nodes > len(args.hosts):
        raise SystemExit(f"num-nodes={args.num_nodes} > available hosts {len(args.hosts)}")
    if args.omp_threads is None:
        args.omp_threads = args.cores_per_numa - args.skip_cores
    args.mbe_order = 2
    # Set atoms_per_node for compat with imported functions
    args.atoms_per_node = args.residues_per_node * 7  # approximate

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    total_residues = args.residues_per_node * args.num_nodes
    _t0 = time.perf_counter()
    mol, n_res = _generate_chain(args)
    _t1 = time.perf_counter()
    print(
        f"target {total_residues} residues -> {mol.natoms} atoms; "
        f"{args.num_nodes} nodes x {args.residues_per_node} residues/node",
        flush=True,
    )
    print(f"generated {mol.natoms} atoms "
          f"(C={mol.elements.count('C') if isinstance(mol.elements, list) else int((np.asarray(mol.elements)=='C').sum())} "
          f"H={int((np.asarray(mol.elements)=='H').sum())} "
          f"N={int((np.asarray(mol.elements)=='N').sum())} "
          f"O={int((np.asarray(mol.elements)=='O').sum())})",
        flush=True,
    )

    schedule, block_mols, cap_mols, dimer_mols = _build_fragments(mol, n_res, args)
    _t2 = time.perf_counter()
    _write_inputs(out, mol, block_mols, cap_mols, dimer_mols, args)
    _t3 = time.perf_counter()
    _write_launch_artifacts(out, schedule, args)
    _t4 = time.perf_counter()
    print(
        f"[gen-time] chain={_t1-_t0:.2f}s  "
        f"fragments={_t2-_t1:.2f}s  "
        f"write_inputs={_t3-_t2:.2f}s  "
        f"write_artifacts={_t4-_t3:.2f}s  "
        f"total={_t4-_t0:.2f}s",
        flush=True,
    )

    print(f"\n{args.num_nodes} blocks (capped, MFCC-style):", flush=True)
    for block in schedule["blocks"]:
        print(
            f"  block {block['block_id']}: host {block['host']} | "
            f"{block['natoms']} atoms",
            flush=True,
        )
    print(f"{len(cap_mols)} conjugate caps (H2)", flush=True)
    if dimer_mols:
        print(f"{len(dimer_mols)} joined dimers (MBE(2)):", flush=True)
        for d in schedule["dimers"]:
            print(
                f"  dimer {d['dimer_id']}: cut {d['cut']} (blocks "
                f"{d['left_block']}+{d['right_block']}) | host {d['host']} | "
                f"{d['natoms']} atoms",
                flush=True,
        )
    print(f"full-chain baseline -> {out / 'full'}", flush=True)
    print(f"\nschedule                -> {out / 'schedule.json'}")
    print(f"recommended launcher    -> {out / 'submit_per_node_local.sh'}")
    print(f"result combiner         -> {out / 'combine_results.py'}")


if __name__ == "__main__":
    main()
