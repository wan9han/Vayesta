#!/usr/bin/env python3
"""Generate whole-PE molecules for ORIGINAL SIESTA NTPoly weak-scaling.

This is NOT the MFCC partition (weak_scale_pe.py). Here each group is ONE
whole polyethylene molecule solved by plain SIESTA/ELSI-NTPoly across all
ranks. For weak scaling: total atoms ~ atoms_per_node * nodes, so each node's
share (~atoms_per_node) stays constant while nodes and the matrix grow.

  group  nodes  carbons  atoms      matrix(N_orb, SZ ~2/atom)  ranks(16/node)
  ws_1   1      1333     4001       ~2668                      16
  ws_4   4      5333     16001      ~10667                     64
  ws_8   8      10666    32000      ~21334                     128

Run each on its node count: mpirun -np 16*nodes siesta < ws_Xnode/input.fdf
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from energy_first.molecule import parse_gen_fdf_text, write_siesta_fdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atoms-per-node", type=int, default=4000,
                    help="target atoms per node (single-node max ~4000)")
    ap.add_argument("--nodes", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--gen-script", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--pseudo-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--basis", default="SZ")
    ap.add_argument("--mesh-cutoff-ry", type=float, default=100.0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Weak-scaling groups (plain SIESTA NTPoly, whole molecule): "
          f"{args.atoms_per_node} atoms/node\n", flush=True)
    print(f"{'group':>8} {'nodes':>6} {'carbons':>8} {'atoms':>7} "
          f"{'N_orb~':>8} {'ranks':>6}", flush=True)

    for n_nodes in args.nodes:
        total = args.atoms_per_node * n_nodes
        n_c = max(round((total - 2) / 3.0), 2)
        atoms = 3 * n_c + 2
        p = subprocess.run([args.python, args.gen_script, str(n_c)],
                           capture_output=True, text=True, check=True)
        mol = parse_gen_fdf_text(p.stdout, label=f"PE_{n_c}C")
        d = out / f"ws_{n_nodes}node"
        d.mkdir(parents=True, exist_ok=True)
        write_siesta_fdf(mol, d / "input.fdf", basis_size=args.basis,
                         mesh_cutoff_ry=args.mesh_cutoff_ry, solution_method="ntpoly")
        for el in set(mol.elements):
            shutil.copy2(Path(args.pseudo_dir) / f"{el}.psf", d / f"{el}.psf")
        n_orb = mol.elements.count("C") * 4 + mol.elements.count("H") * 1  # SZ approx
        print(f"ws_{n_nodes}node {n_nodes:>6} {n_c:>8} {atoms:>7} {n_orb:>8} "
              f"{16 * n_nodes:>6}  -> {d}/input.fdf", flush=True)

    print("\nRun (example, 4 nodes):", flush=True)
    print(f"  cd {out}/ws_4node && mpirun -np 64 -hostfile <hosts> "
          f"-x OMP_NUM_THREADS=36 <siesta> < input.fdf > siesta.out", flush=True)
    print("\nWeak scaling: wall time should stay ~flat across ws_1 -> ws_4 -> ws_8.", flush=True)


if __name__ == "__main__":
    main()
