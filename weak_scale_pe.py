#!/usr/bin/env python3
"""Weak-scaling PE generator + per-node block partitioner for HONPAS.

Target platform: intranet-optimized HONPAS, 16 NUMA/node, 1 MPI proc/NUMA
-> 16 procs/node, up to 16 nodes.

Model (codex.md weak scaling):
  * total chain length is set by  atoms_per_node * num_nodes  (weak scaling:
    each node gets the same ~atoms_per_node regardless of node count);
  * the chain is split into num_nodes CONTIGUOUS blocks, one per node;
  * each block is capped with H at its cut ends (MFCC-style) so it is a valid
    closed-shell molecule that SIESTA can solve standalone;
  * block b is assigned to MPI ranks [procs_per_node*b .. procs_per_node*(b+1)-1]
    on node b -> those ranks solve that one block together (intra-node comm).

Outputs (under --out-dir):
  block_NNNN/input.fdf    one capped PE block, NTPoly (for 16-rank solve)
  cap_NNNN/input.fdf      conjugate caps (for MFCC energy combination)
  schedule.json           block<->rank<->node mapping + sizes
  launch.sh               per-node concurrent mpirun -np 16, then MFCC combine

Example:
  python weak_scale_pe.py --atoms-per-node 5000 --num-nodes 4 \
      --gen-script .../gen.py --pseudo-dir .../testcases --out-dir ./ws4
  -> ~20002-atom PE chain, 4 blocks (~5000 atoms each), ranks 0-15/16-31/32-47/48-63.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from energy_first.molecule import Molecule, parse_gen_fdf_text, write_siesta_fdf

CH_BOND = 1.10  # C-H cap bond length (Angstrom)


def _cap_pos(c_coord, neighbor_coord, ch=CH_BOND):
    """Cap-H coordinate on carbon c, pointing toward its (removed) neighbor."""
    v = np.asarray(neighbor_coord) - np.asarray(c_coord)
    d = np.linalg.norm(v)
    if d == 0:
        raise ValueError("cut carbons coincide")
    return np.asarray(c_coord) + v / d * ch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atoms-per-node", type=int, required=True,
                    help="target atoms per node (PE ~3 atoms/C; per-block atom count reported)")
    ap.add_argument("--num-nodes", type=int, required=True, help="number of nodes = number of blocks")
    ap.add_argument("--procs-per-node", type=int, default=16, help="MPI procs per node (1/NUMA)")
    ap.add_argument("--gen-script", required=True, help="testcases/gen.py")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--pseudo-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--basis", default="SZ")
    ap.add_argument("--mesh-cutoff-ry", type=float, default=100.0)
    # ---- HONPAS binding (no SLURM; mpirun from first host) ----
    ap.add_argument("--hosts", nargs="+",
                    default=["71.20.27.21", "71.20.27.22", "71.20.27.23", "71.20.27.24",
                             "71.20.27.33", "71.20.27.34", "71.20.27.35", "71.20.27.36"],
                    help="available machines; first num-nodes are used")
    ap.add_argument("--num-numa", type=int, default=16, help="NUMAs per node")
    ap.add_argument("--cores-per-numa", type=int, default=38, help="cores per NUMA")
    ap.add_argument("--skip-cores", type=int, default=2,
                    help="skip this many first cores of EACH NUMA (use the rest for 1 MPI proc)")
    ap.add_argument("--omp-threads", type=int, default=None,
                    help="OMP/MKL/OPENBLAS threads per MPI proc (default = cores_per_numa - skip_cores)")
    args = ap.parse_args()
    if args.num_nodes > len(args.hosts):
        raise SystemExit(f"num-nodes={args.num_nodes} > available hosts {len(args.hosts)}")
    if args.omp_threads is None:
        args.omp_threads = args.cores_per_numa - args.skip_cores

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    nn, ppn = args.num_nodes, args.procs_per_node

    # ---- 1. total chain length: PE Cn H2n+2 = 3n+2 atoms ~ atoms_per_node*num_nodes
    target = args.atoms_per_node * nn
    n_c = max(round((target - 2) / 3.0), nn * 2)
    real_atoms = 3 * n_c + 2
    print(f"target {target} atoms -> C{n_c}H{2*n_c+2} = {real_atoms} atoms "
          f"(+head/tail H included); {nn} nodes x ~{args.atoms_per_node} atoms/node", flush=True)

    # ---- 2. generate full chain
    p = subprocess.run([args.python, args.gen_script, str(n_c)],
                       capture_output=True, text=True, check=True)
    mol = parse_gen_fdf_text(p.stdout, label=f"PE_{n_c}C")
    coords = mol.coords
    print(f"generated {mol.natoms} atoms", flush=True)

    # ---- 3. carbons in chain order; vectorized H->nearest-carbon assignment
    cs = [i for i, e in enumerate(mol.elements) if e == "C"]
    cs.sort(key=lambda i: (coords[i, 0], coords[i, 1], coords[i, 2]))  # along chain axis
    cpos = coords[cs]
    h_idx = [i for i, e in enumerate(mol.elements) if e == "H"]
    h_to_c = {}  # H atom -> carbon atom index
    for h in h_idx:
        d = np.linalg.norm(cpos - coords[h], axis=1)
        h_to_c[h] = cs[int(np.argmin(d))]

    # ---- 4. partition carbons into nn contiguous blocks (near-equal)
    edges = [round(k * len(cs) / nn) for k in range(nn + 1)]  # carbon-index ranges

    def carbon_block(b):
        return cs[edges[b]:edges[b + 1]]

    def cut_pairs():
        # cut between block b-1 last carbon and block b first carbon
        pairs = []
        for b in range(1, nn):
            left = carbon_block(b - 1)[-1]
            right = carbon_block(b)[0]
            pairs.append((left, right))
        return pairs

    cuts = cut_pairs()

    # ---- 5. build each capped block molecule + conjugate caps
    schedule = {"atoms_per_node_target": args.atoms_per_node, "num_nodes": nn,
                "procs_per_node": ppn, "total_chain_atoms": mol.natoms,
                "carbons_total": n_c, "blocks": [], "caps": [], "cuts": cuts}
    block_mols = []
    cap_mols = []
    cap_pos_per_cut = []  # for each cut: (capH_on_left, capH_on_right)

    for b in range(nn):
        carbons = carbon_block(b)
        cset = set(carbons)
        atoms = list(carbons) + [h for h in h_idx if h_to_c[h] in cset]
        els = [mol.elements[a] for a in atoms]
        co = coords[atoms].tolist()
        # cap left end if cut
        cap_left = None
        if b > 0:
            c0 = carbons[0]
            prev = cs[edges[b] - 1]  # last carbon of previous block
            cap_left = _cap_pos(coords[c0], coords[prev])
        cap_right = None
        if b < nn - 1:
            cN = carbons[-1]
            nxt = cs[edges[b + 1]]  # first carbon of next block
            cap_right = _cap_pos(coords[cN], coords[nxt])
        # build molecule
        m = Molecule(list(els), np.array(co, dtype=float), label=f"block_{b:04d}")
        if cap_left is not None:
            m.append("H", cap_left)
        if cap_right is not None:
            m.append("H", cap_right)
        block_mols.append(m)
        rank0, rank1 = b * ppn, (b + 1) * ppn - 1
        schedule["blocks"].append({
            "block_id": b, "node_id": b, "ranks": list(range(rank0, rank1 + 1)),
            "natoms": m.natoms, "carbons": len(carbons),
            "caps_added": (1 if cap_left is not None else 0) + (1 if cap_right is not None else 0),
        })

    # conjugate cap per cut: H2 at the two cap-H positions (consistency -> cancellation)
    for (left_c, right_c) in cuts:
        # cap on left_c points toward right_c; cap on right_c points toward left_c
        chL = _cap_pos(coords[left_c], coords[right_c])
        chR = _cap_pos(coords[right_c], coords[left_c])
        cap_mols.append(Molecule(["H", "H"], np.vstack([chL, chR]), label=f"cap_{len(cap_mols):04d}"))
        cap_pos_per_cut.append((chL, chR))

    # ---- 6. write FDFs + pseudos
    for b, m in enumerate(block_mols):
        bd = out / f"block_{b:04d}"
        bd.mkdir(parents=True, exist_ok=True)
        write_siesta_fdf(m, bd / "input.fdf", basis_size=args.basis,
                         mesh_cutoff_ry=args.mesh_cutoff_ry, solution_method="ntpoly")
        for el in set(m.elements):
            shutil.copy2(Path(args.pseudo_dir) / f"{el}.psf", bd / f"{el}.psf")
    for c, m in enumerate(cap_mols):
        cd = out / f"cap_{c:04d}"
        cd.mkdir(exist_ok=True)
        write_siesta_fdf(m, cd / "input.fdf", basis_size=args.basis,
                         mesh_cutoff_ry=args.mesh_cutoff_ry, solution_method="ntpoly")
        for el in set(m.elements):
            shutil.copy2(Path(args.pseudo_dir) / f"{el}.psf", cd / f"{el}.psf")

    # ---- 7. schedule + rankfile + bound launch.sh (HONPAS, no SLURM)
    (out / "schedule.json").write_text(json.dumps(schedule, indent=2))

    # Per-NUMA core ranges: NUMA k -> global cores [k*cpn+skip .. k*cpn+cpn-1].
    # ASSUMES contiguous NUMA layout (NUMA k occupies cores k*cpn .. k*cpn+cpn-1).
    slots = []
    for k in range(args.num_numa):
        lo = k * args.cores_per_numa + args.skip_cores
        hi = k * args.cores_per_numa + args.cores_per_numa - 1
        slots.append(f"{lo}-{hi}")
    rf = ["# OpenMPI rankfile: rank r -> NUMA r -> that NUMA's cores (skip first skip_cores).",
          "# __HOST__ is substituted per target node by launch.sh.",
          "# VERIFY the NUMA->CPU map with `numactl -H` / `lscpu -e`; edit if topology differs."]
    for k, s in enumerate(slots):
        rf.append(f"rank {k}=__HOST__ slot={s}")
    (out / "rankfile_template.txt").write_text("\n".join(rf) + "\n")

    hosts = args.hosts[:nn]
    omp = args.omp_threads
    launch = []
    launch.append("#!/bin/bash")
    launch.append("# HONPAS weak-scaling launch (no SLURM). Run from the first host via mpirun --host.")
    launch.append(f"# {nn} blocks x {args.num_numa} MPI ranks/block; 1 rank/NUMA, {omp} OMP threads/rank.")
    launch.append("# block b -> node hosts[b]; ranks bound by rankfile (1 per NUMA, cores skip first 2).")
    launch.append("# !!! ASSUMES NUMA k = cores [k*38 .. k*38+37]. VERIFY with `numactl -H`; fix rankfile_*.txt if not.")
    launch.append("# !!! Requires passwordless ssh from this host to every host in HOSTS.")
    launch.append(f'HOSTS=({" ".join(hosts)})')
    launch.append('N="${1:-' + str(nn) + '}"  # override node count: ./launch.sh [N] (regenerate if N>nn)')
    launch.append('SIESTA="${SIESTA:-siesta}"')
    launch.append(f'export OMP_NUM_THREADS={omp} MKL_NUM_THREADS={omp} OPENBLAS_NUM_THREADS={omp}')
    launch.append('export OMP_PROC_BIND=close OMP_PLACES=cores')
    launch.append(f'cd "{out}"')
    launch.append("for ((i=0;i<N;i++)); do")
    launch.append('  h=${HOSTS[$i]}')
    launch.append('  blk=$(printf "block_%04d" $i)')
    launch.append('  sed "s/__HOST__/$h/" rankfile_template.txt > "$blk/rankfile.txt"')
    launch.append(f'  ( cd "$blk" && mpirun -np {args.num_numa} --host "$h" \\')
    launch.append('      --rankfile rankfile.txt --bind-to core --report-bindings \\')
    launch.append('      -x OMP_NUM_THREADS -x MKL_NUM_THREADS -x OPENBLAS_NUM_THREADS \\')
    launch.append('      -x OMP_PROC_BIND -x OMP_PLACES \\')
    launch.append('      "$SIESTA" < input.fdf > siesta.out 2>&1 ) &')
    launch.append("done")
    launch.append("wait")
    launch.append('echo "all blocks done; combining MFCC E^(1) = sum(blocks) - sum(caps)"')
    combine = (
        f'{args.python} -c "'
        f"import re;"
        f"def E(p):"
        f"  t=open(p+'/siesta.out').read();"
        f"  m=re.findall(r'Total\\\\s*=\\\\s*(-?\\d+\\.\\d+)',t);"
        f"  return float(m[-1]) if m else None;"
        f"bs=[E('block_%04d'%b) for b in range({nn})];"
        f"cs=[E('cap_%04d'%c) for c in range({nn-1})];"
        f"print('blocks',bs); print('caps',cs);"
        f"print('E_MBE1 =', sum(x for x in bs if x is not None)-sum(x for x in cs if x is not None))"
        f'"'
    )
    launch.append(combine)
    (out / "launch.sh").write_text("\n".join(launch) + "\n")
    os.chmod(out / "launch.sh", 0o755)

    # ---- 8. report
    print(f"\n{nn} blocks (capped, MFCC-style):", flush=True)
    for b, m in enumerate(block_mols):
        s = schedule["blocks"][b]
        print(f"  block {b}: ranks {s['ranks'][0]}-{s['ranks'][-1]} (node {b}) | "
              f"{m.natoms} atoms, {s['carbons']} C, {s['caps_added']} cap H", flush=True)
    print(f"{len(cap_mols)} conjugate caps (H2 each)", flush=True)
    print(f"\nschedule -> {out/'schedule.json'}")
    print(f"launch   -> {out/'launch.sh'}")


if __name__ == "__main__":
    main()
