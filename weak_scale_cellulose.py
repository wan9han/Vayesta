#!/usr/bin/env python3
"""Weak-scaling cellulose I-beta generator (TRS2 + MBE(2)).

Mirrors weak_scale_pe.py / weak_scale_peg.py but for clean Nishiyama cellulose
Ibeta (energy_first/build_cellulose_ibeta.py: 2_1 helix, no tiling strain, 4.2 eV
gap stable with length -> TRS2 converges cleanly). MFCC cuts the C-O glycosidic
(beta-1,4) bonds; reuses weak_scale_pe's launch/combine infrastructure.

PE-parity scale: --glucose-per-node 296 (~15990 SZ orbitals ~ PE's 16000-orbital
per-node matrix, 4000x4000 per rank at 16 ranks).

Usage:
  python3 weak_scale_cellulose.py --glucose-per-node 296 --num-nodes 4 --out-dir /tmp/ws_cellu4
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from weak_scale_pe import (
    DEFAULT_HOSTS, _slot_ranges, _place_pseudo, _write_inputs, _write_launch_artifacts,
)
from energy_first.molecule import Molecule
from energy_first.build_cellulose_ibeta import build_chain
from cellulose_validate import find_glycosidic_bonds, fragment_cellulose, build_dimer_cellulose


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glucose-per-node", type=int, required=True,
                    help="glucose units per node (~54 SZ orbitals each; 296 ~ PE parity 16000 orb)")
    ap.add_argument("--num-nodes", type=int, required=True)
    ap.add_argument("--procs-per-node", type=int, default=16)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--pseudo-dir", default=str(Path(__file__).resolve().parent / "pseudos"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--basis", default="SZ")
    ap.add_argument("--mesh-cutoff-ry", type=float, default=100.0)
    ap.add_argument("--block-slice-num", type=int, default=1)
    ap.add_argument("--dimer-slice-num", type=int, default=2)
    ap.add_argument("--shared-pseudo", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--full-baseline", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--solution-method", default="ntpoly", choices=["ntpoly", "diagonali"],
                    help="TRS2 (ntpoly) default -- cellulose's 4.2 eV gap converges cleanly.")
    ap.add_argument("--full-solution-method", default="ntpoly", choices=["ntpoly", "diagonali"])
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


def _generate_chain(total_glucose):
    atoms, bonds = build_chain(total_glucose)
    el = [a.element for a in atoms]
    co = np.array([a.xyz for a in atoms])
    return Molecule(el, co, f"cellu{total_glucose}")


def _build_fragments(mol, total_glucose, args):
    bonds = find_glycosidic_bonds(mol)
    num_nodes = args.num_nodes
    if num_nodes == 1:
        cuts = []
    else:
        n = len(bonds)
        cuts = [bonds[(k * n) // num_nodes] for k in range(1, num_nodes)]
    frag_mols, cap_mols, comp = fragment_cellulose(mol, cuts)
    dimer_mols = []
    if cuts:
        for k, cut in enumerate(cuts):
            d = build_dimer_cellulose(mol, cuts, cut)
            d.label = f"dimer_{k:04d}"
            dimer_mols.append(d)
    for i, f in enumerate(frag_mols):
        f.label = f"block_{i:04d}"
    for i, c in enumerate(cap_mols):
        c.label = f"cap_{i:04d}"

    schedule = {
        "glucose_per_node": args.glucose_per_node,
        "num_nodes": num_nodes,
        "procs_per_node": args.procs_per_node,
        "num_numa": args.num_numa,
        "omp_threads": args.omp_threads,
        "total_chain_atoms": mol.natoms,
        "total_glucose": total_glucose,
        "hosts": args.hosts[:num_nodes],
        "cuts": [list(c) for c in cuts],
        "blocks": [], "caps": [], "dimers": [],
        "mbe_order": 2, "system": "cellulose-Ibeta",
    }
    for i, f in enumerate(frag_mols):
        schedule["blocks"].append({"block_id": i, "host": args.hosts[i], "natoms": f.natoms})
    for i, c in enumerate(cap_mols):
        schedule["caps"].append({"cap_id": i, "natoms": c.natoms, "host": args.hosts[0]})
    for k, d in enumerate(dimer_mols):
        schedule["dimers"].append({
            "dimer_id": k, "cut": list(cuts[k]),
            "left_block": k, "right_block": k + 1,
            "host": args.hosts[k % num_nodes], "natoms": d.natoms,
        })
    return schedule, frag_mols, cap_mols, dimer_mols


def main():
    args = _parse_args()
    if args.num_nodes > len(args.hosts):
        raise SystemExit(f"num-nodes={args.num_nodes} > hosts {len(args.hosts)}")
    if args.omp_threads is None:
        args.omp_threads = args.cores_per_numa - args.skip_cores
    args.mbe_order = 2
    args.atoms_per_node = args.glucose_per_node * 21     # compat with imported helpers
    args.in_process_gen = True
    args.gen_script = "build_cellulose_ibeta.py (in-process)"

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    total = args.glucose_per_node * args.num_nodes
    t0 = time.perf_counter()
    mol = _generate_chain(total)
    t1 = time.perf_counter()
    el = np.asarray(mol.elements)
    print(f"target {total} glucose -> {mol.natoms} atoms; "
          f"{args.num_nodes} nodes x {args.glucose_per_node} glu/node", flush=True)
    print(f"generated (C/H/O={int((el=='C').sum())}/{int((el=='H').sum())}/{int((el=='O').sum())})",
          flush=True)

    schedule, block_mols, cap_mols, dimer_mols = _build_fragments(mol, total, args)
    t2 = time.perf_counter()
    _write_inputs(out, mol, block_mols, cap_mols, dimer_mols, args)
    t3 = time.perf_counter()
    _write_launch_artifacts(out, schedule, args)
    t4 = time.perf_counter()
    print(f"[gen-time] chain={t1-t0:.2f}s fragments={t2-t1:.2f}s "
          f"write_inputs={t3-t2:.2f}s artifacts={t4-t3:.2f}s total={t4-t0:.2f}s", flush=True)

    print(f"\n{args.num_nodes} blocks (capped, MFCC-style):", flush=True)
    for b in schedule["blocks"]:
        print(f"  block {b['block_id']}: host {b['host']} | {b['natoms']} atoms", flush=True)
    print(f"{len(cap_mols)} conjugate caps (H2)", flush=True)
    if dimer_mols:
        print(f"{len(dimer_mols)} joined dimers (MBE(2)):", flush=True)
        for d in schedule["dimers"]:
            print(f"  dimer {d['dimer_id']}: cut {d['cut']} (blocks "
                  f"{d['left_block']}+{d['right_block']}) | {d['natoms']} atoms", flush=True)
    print(f"full-chain baseline -> {out / 'full'}")
    print(f"\nschedule             -> {out / 'schedule.json'}")
    print(f"recommended launcher -> {out / 'submit_per_node_local.sh'}")
    print(f"result combiner      -> {out / 'combine_results.py'}")


if __name__ == "__main__":
    main()
