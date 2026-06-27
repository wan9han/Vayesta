#!/usr/bin/env python3
"""Weak-scaling PE generator + HONPAS launch scaffolding.

The original version of this script assumed one "head-node mpirun" that
launched every block remotely. The new default keeps that path as a legacy
artifact, but also generates the more realistic HONPAS flow:

* one capped PE block per target node;
* each node runs its own local ``mpirun`` for that block only;
* a coordinator script ``ssh``'s into each host and triggers the local job;
* MFCC conjugate caps are run separately (small serial jobs);
* **MBE(2) joined dimers** (one per cut, ~2x a block) restore the per-cut
  interaction that plain MFCC misses (~1.24 eV/cut H-cap error). Without them
  the combiner falls back to MFCC(1) and the total is off by ~1.24 eV/cut.

Energy combination (``combine_results.py``):

    E^(1) = Σ E(block) − Σ E(cap)                         # MFCC(1), ~1.24 eV/cut
    E^(2) = E^(1) + Σ_c [E(dimer_c) − E(block) − E(cap)]  # MBE(2), ≈ E_full

The cap terms cancel over all cuts, so E^(2) is inclusion–exclusion over the
overlapping joined pieces. E^(2) reproduces E_full exactly for 2 fragments and
to the small 3-body residual (~1e-3 eV/cut) for a long chain. Cost: N−1 extra
dimer jobs at ~2x the block size; each is still a single bounded node-local
solve, so the weak-scaling structure is preserved.

That structure matches the weak-scaling claim more honestly: the main timing
signal is the per-node block solve, not a single distributed launcher.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from energy_first.molecule import Molecule, parse_gen_fdf_text, write_siesta_fdf

CH_BOND = 1.10
DEFAULT_HOSTS = [
    "71.20.27.21",
    "71.20.27.22",
    "71.20.27.23",
    "71.20.27.24",
    "71.20.27.33",
    "71.20.27.34",
    "71.20.27.35",
    "71.20.27.36",
    "71.20.16.12",
    "71.20.16.22",
    "71.20.16.32",
    "71.20.16.42",
    "71.20.16.132",
    "71.20.16.142",
    "71.20.16.152",
    "71.20.16.162",
]
DEFAULT_GEN_SCRIPT = Path(__file__).resolve().parent / "gen.py"


def _cap_pos(c_coord, neighbor_coord, ch=CH_BOND):
    """Cap-H coordinate on carbon c, pointing toward its removed neighbor."""
    v = np.asarray(neighbor_coord) - np.asarray(c_coord)
    d = np.linalg.norm(v)
    if d == 0:
        raise ValueError("cut carbons coincide")
    return np.asarray(c_coord) + v / d * ch


def _write(path: Path, text: str, mode: int | None = None) -> None:
    path.write_text(text)
    if mode is not None:
        os.chmod(path, mode)


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--atoms-per-node",
        type=int,
        required=True,
        help="target atoms per node (PE ~3 atoms/C; per-block atom count reported)",
    )
    ap.add_argument(
        "--num-nodes",
        type=int,
        required=True,
        help="number of nodes = number of capped PE blocks",
    )
    ap.add_argument(
        "--procs-per-node",
        type=int,
        default=16,
        help="MPI ranks per node / per block (1 rank per NUMA by default)",
    )
    ap.add_argument(
        "--gen-script",
        default=str(DEFAULT_GEN_SCRIPT),
        help="path to PE generator script (default: repo-local gen.py)",
    )
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--pseudo-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--basis", default="SZ")
    ap.add_argument("--mesh-cutoff-ry", type=float, default=100.0)
    ap.add_argument(
        "--mbe-order",
        type=int,
        default=2,
        choices=[1, 2],
        help=(
            "energy combination order: 1 = plain MFCC (sum blocks - sum caps, "
            "~1.24 eV/cut cap error); 2 = MFCC + 2-body MBE over joined dimers "
            "(restores the per-cut interaction; default). Order 2 emits one "
            "dimer job per cut."
        ),
    )
    ap.add_argument(
        "--solution-method",
        default="ntpoly",
        choices=["ntpoly", "diagonali"],
        help=(
            "SIESTA density-matrix solver for blocks/dimers/full-baseline "
            "(ntpoly = production sparse solver; diagonali = LAPACK, for small "
            "local verification). Caps are always tiny H2 jobs run with "
            "diagonali."
        ),
    )
    ap.add_argument(
        "--block-slice-num",
        type=int,
        default=1,
        help=(
            "NTPOLY_SLICE_NUM baked into each block (and full-baseline) "
            "run_local.sh. Default 1; raise to split the density-matrix "
            "purification for large blocks."
        ),
    )
    ap.add_argument(
        "--dimer-slice-num",
        type=int,
        default=1,
        help=(
            "NTPOLY_SLICE_NUM baked into each dimer run_local.sh. Default 1; "
            "dimers are ~2x a block so usually warrant a higher value than "
            "blocks at the same per-node atom count."
        ),
    )
    ap.add_argument("--hosts", nargs="+", default=DEFAULT_HOSTS)
    ap.add_argument("--num-numa", type=int, default=16, help="NUMAs per node")
    ap.add_argument("--cores-per-numa", type=int, default=38, help="cores per NUMA")
    ap.add_argument(
        "--skip-cores",
        type=int,
        default=2,
        help="skip this many leading cores in each NUMA domain",
    )
    ap.add_argument(
        "--omp-threads",
        type=int,
        default=None,
        help="OMP/MKL/OPENBLAS threads per MPI rank (default = cores_per_numa - skip_cores)",
    )
    ap.add_argument(
        "--pkg-root",
        default="/share/honpas/xzz/siesta-20260520",
        help="shared package root used on compute nodes",
    )
    ap.add_argument(
        "--mpi-prefix",
        default="/share/hmpi2.4.1/hmpi-v2.4.1-huawei",
        help="MPI installation prefix used on compute nodes",
    )
    ap.add_argument(
        "--siesta-app",
        default=None,
        help="SIESTA executable path on compute nodes (default derived from pkg-root)",
    )
    ap.add_argument(
        "--env-sh",
        default=None,
        help="env.sh to source on compute nodes (default derived from pkg-root)",
    )
    ap.add_argument(
        "--launch-orted",
        default=None,
        help="path to launch_orted.sh for legacy multi-node mpirun",
    )
    ap.add_argument(
        "--remote-out-dir",
        default=None,
        help="shared path visible from compute nodes; defaults to absolute out-dir path",
    )
    ap.add_argument(
        "--ssh-user",
        default="",
        help="optional user for ssh launch, e.g. xzz",
    )
    return ap.parse_args()


def _generate_chain(args):
    target = args.atoms_per_node * args.num_nodes
    n_c = max(round((target - 2) / 3.0), args.num_nodes * 2)
    proc = subprocess.run(
        [args.python, args.gen_script, str(n_c)],
        capture_output=True,
        text=True,
        check=True,
    )
    mol = parse_gen_fdf_text(proc.stdout, label=f"PE_{n_c}C")
    return mol, n_c


def _build_fragments(mol: Molecule, n_c: int, args):
    coords = mol.coords
    cs = [i for i, e in enumerate(mol.elements) if e == "C"]
    cs.sort(key=lambda i: (coords[i, 0], coords[i, 1], coords[i, 2]))
    cpos = coords[cs]
    h_idx = [i for i, e in enumerate(mol.elements) if e == "H"]
    # Map each H to its nearest C. The original brute-force was O(n_H * n_C) = O(N^2)
    # (a Python loop, one all-carbon norm per H), which dominates system generation
    # and becomes infeasible at E-scale. cKDTree gives the SAME nearest-carbon result
    # (query k=1 == argmin over Euclidean norm; C-H bonds are unique, no ties) in
    # O(N log N). See 生成优化文档.md.
    h_to_c = {}
    if h_idx:
        from scipy.spatial import cKDTree
        tree = cKDTree(cpos)
        _, nn = tree.query(coords[h_idx], k=1)
        nn = np.atleast_1d(nn)
        for h, j in zip(h_idx, nn):
            h_to_c[h] = cs[int(j)]

    edges = [round(k * len(cs) / args.num_nodes) for k in range(args.num_nodes + 1)]

    def carbon_block(b):
        return cs[edges[b] : edges[b + 1]]

    cuts = []
    for b in range(1, args.num_nodes):
        left = carbon_block(b - 1)[-1]
        right = carbon_block(b)[0]
        cuts.append((left, right))

    schedule = {
        "atoms_per_node_target": args.atoms_per_node,
        "num_nodes": args.num_nodes,
        "procs_per_node": args.procs_per_node,
        "num_numa": args.num_numa,
        "omp_threads": args.omp_threads,
        "total_chain_atoms": mol.natoms,
        "carbons_total": n_c,
        "hosts": args.hosts[: args.num_nodes],
        "cuts": cuts,
        "blocks": [],
        "caps": [],
        "dimers": [],
        "mbe_order": getattr(args, "mbe_order", 2),
    }

    block_mols = []
    for b in range(args.num_nodes):
        carbons = carbon_block(b)
        cset = set(carbons)
        atoms = list(carbons) + [h for h in h_idx if h_to_c[h] in cset]
        els = [mol.elements[a] for a in atoms]
        co = coords[atoms].tolist()
        cap_left = None
        if b > 0:
            c0 = carbons[0]
            prev = cs[edges[b] - 1]
            cap_left = _cap_pos(coords[c0], coords[prev])
        cap_right = None
        if b < args.num_nodes - 1:
            cN = carbons[-1]
            nxt = cs[edges[b + 1]]
            cap_right = _cap_pos(coords[cN], coords[nxt])
        m = Molecule(list(els), np.array(co, dtype=float), label=f"block_{b:04d}")
        if cap_left is not None:
            m.append("H", cap_left)
        if cap_right is not None:
            m.append("H", cap_right)
        block_mols.append(m)
        schedule["blocks"].append(
            {
                "block_id": b,
                "host": args.hosts[b],
                "natoms": m.natoms,
                "carbons": len(carbons),
                "caps_added": int(cap_left is not None) + int(cap_right is not None),
            }
        )

    cap_mols = []
    for idx, (left_c, right_c) in enumerate(cuts):
        ch_l = _cap_pos(coords[left_c], coords[right_c])
        ch_r = _cap_pos(coords[right_c], coords[left_c])
        cap = Molecule(["H", "H"], np.vstack([ch_l, ch_r]), label=f"cap_{idx:04d}")
        cap_mols.append(cap)
        schedule["caps"].append({"cap_id": idx, "natoms": cap.natoms, "host": args.hosts[0]})

    # Joined dimers for the MBE(2) correction. Cut k joins block k and block k+1
    # (real C-C bond at cut k restored, capped only at the *outer* boundaries —
    # the same cap placement those two blocks already carry). The per-cut
    # increment E(dimer_k) - E(block_k) - E(block_{k+1}) + E(cap_k) restores the
    # interaction MFCC(1) misses; the cap terms cancel over all cuts, leaving
    # E^(2) = Σ E(block) + Σ_k [E(dimer_k) - E(block_k) - E(block_{k+1})].
    # Without these, the combiner falls back to plain MFCC(1) (~1.24 eV/cut).
    dimer_mols = []
    if getattr(args, "mbe_order", 2) >= 2:
        for k in range(args.num_nodes - 1):
            carbons = cs[edges[k] : edges[k + 2]]
            cset = set(carbons)
            atoms = list(carbons) + [h for h in h_idx if h_to_c[h] in cset]
            els = [mol.elements[a] for a in atoms]
            co = coords[atoms].tolist()
            m = Molecule(list(els), np.array(co, dtype=float), label=f"dimer_{k:04d}")
            caps_added = 0
            if k > 0:  # left outer boundary = cut k-1
                c0 = carbons[0]
                prev = cs[edges[k] - 1]
                m.append("H", _cap_pos(coords[c0], coords[prev]))
                caps_added += 1
            if k < args.num_nodes - 2:  # right outer boundary = cut k+1
                cN = carbons[-1]
                nxt = cs[edges[k + 2]]
                m.append("H", _cap_pos(coords[cN], coords[nxt]))
                caps_added += 1
            dimer_mols.append(m)
            schedule["dimers"].append(
                {
                    "dimer_id": k,
                    "cut": list(cuts[k]),
                    "left_block": k,
                    "right_block": k + 1,
                    "host": args.hosts[k % args.num_nodes],
                    "natoms": m.natoms,
                    "carbons": len(carbons),
                    "caps_added": caps_added,
                }
            )

    return schedule, block_mols, cap_mols, dimer_mols


def _slot_ranges(args):
    slots = []
    for k in range(args.num_numa):
        lo = k * args.cores_per_numa + args.skip_cores
        hi = k * args.cores_per_numa + args.cores_per_numa - 1
        slots.append((lo, hi))
    return slots


def _write_inputs(out: Path, full_mol, block_mols, cap_mols, dimer_mols, args):
    pseudo_dir = Path(args.pseudo_dir)
    solver = getattr(args, "solution_method", "ntpoly")
    for b, mol in enumerate(block_mols):
        block_dir = out / f"block_{b:04d}"
        block_dir.mkdir(parents=True, exist_ok=True)
        write_siesta_fdf(
            mol,
            block_dir / "input.fdf",
            basis_size=args.basis,
            mesh_cutoff_ry=args.mesh_cutoff_ry,
            solution_method=solver,
        )
        for el in set(mol.elements):
            shutil.copy2(pseudo_dir / f"{el}.psf", block_dir / f"{el}.psf")

    # Joined dimers (~2x a block): same sparse solver as the blocks so the
    # dimer energy is directly comparable in the MBE(2) sum.
    for k, mol in enumerate(dimer_mols):
        dimer_dir = out / f"dimer_{k:04d}"
        dimer_dir.mkdir(parents=True, exist_ok=True)
        write_siesta_fdf(
            mol,
            dimer_dir / "input.fdf",
            basis_size=args.basis,
            mesh_cutoff_ry=args.mesh_cutoff_ry,
            solution_method=solver,
        )
        for el in set(mol.elements):
            shutil.copy2(pseudo_dir / f"{el}.psf", dimer_dir / f"{el}.psf")

    for c, mol in enumerate(cap_mols):
        cap_dir = out / f"cap_{c:04d}"
        cap_dir.mkdir(parents=True, exist_ok=True)
        write_siesta_fdf(
            mol,
            cap_dir / "input.fdf",
            basis_size=args.basis,
            mesh_cutoff_ry=args.mesh_cutoff_ry,
            solution_method="diagonali",
        )
        for el in set(mol.elements):
            shutil.copy2(pseudo_dir / f"{el}.psf", cap_dir / f"{el}.psf")

    # Unfragmented full-chain baseline (MFCC reference energy). Same numerical
    # settings as the blocks so E_full is directly comparable to E_mfcc/E_mbe2.
    full_dir = out / "full"
    full_dir.mkdir(parents=True, exist_ok=True)
    write_siesta_fdf(
        full_mol,
        full_dir / "input.fdf",
        basis_size=args.basis,
        mesh_cutoff_ry=args.mesh_cutoff_ry,
        solution_method=solver,
    )
    for el in set(full_mol.elements):
        shutil.copy2(pseudo_dir / f"{el}.psf", full_dir / f"{el}.psf")


def _render_env_script(args):
    app = args.siesta_app or f"{args.pkg_root}/siesta/build-clang/Src/siesta"
    env_sh = args.env_sh or f"{args.pkg_root}/env.sh"
    mpi_prefix = args.mpi_prefix
    return f"""#!/bin/bash
set -euo pipefail

export PKG_ROOT="${{PKG_ROOT:-{args.pkg_root}}}"
export MPI_PREFIX="${{MPI_PREFIX:-{mpi_prefix}}}"
export APP="${{APP:-{app}}}"
export ENV_SH="${{ENV_SH:-{env_sh}}}"
export MPIRUN="${{MPIRUN:-${{MPI_PREFIX}}/bin/mpirun}}"
export XPMEM_HOME="${{XPMEM_HOME:-/share/hmpi2.4.1/xpmem-2.7.3}}"
export UCX_HOME="${{UCX_HOME:-/share/hmpi2.4.1/hucx-v2.4.1-huawei}}"
export UCG_HOME="${{UCG_HOME:-/share/hmpi2.4.1/xucg-v2.4.1-huawei}}"
export MPI_HOME="${{MPI_HOME:-${{MPI_PREFIX}}}}"
export BISHENG_LIB="${{BISHENG_LIB:-/share/honpas/xzz/siesta-20260520/BiShengCompiler-4.2.0.2-aarch64-linux/lib:/share/honpas/xzz/siesta-20260520/BiShengCompiler-4.2.0.2-aarch64-linux/lib64}}"
export BISHENG_PATH="${{BISHENG_PATH:-/share/honpas/xzz/siesta-20260520/BiShengCompiler-4.2.0.2-aarch64-linux/bin}}"

source "${{ENV_SH}}"

export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export OPAL_PREFIX="${{MPI_PREFIX}}"
export CORE_PER_NUMA={args.cores_per_numa}
export CORE_BIAS={args.skip_cores}
export NUM_NUMAS={args.num_numa}
export p_perdie={args.cores_per_numa - args.skip_cores}
export ndie={args.num_numa}
export OMP_NUM_THREADS={args.omp_threads}
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OPENBLAS_NUM_THREADS={args.omp_threads}
export MKL_NUM_THREADS={args.omp_threads}
export NTPOLY_SLICE_NUM=1
export MEMKIND_HBW_NODES=16-31
export VERBS_LOG_LEVEL=0
export UCX_LOG_LEVEL=info
export UCX_TLS=rc,sm
export NET_DEVICE_ALL="hns_0:1,hns_1:1,hns_3:1,hns_2:1,hns_4:1,hns_5:1,hns_7:1,hns_6:1"
export OMPI_CC=/share/honpas/xzz/siesta-20260520/BiShengCompiler-4.2.0.2-aarch64-linux/bin/clang
export OMPI_CXX=/share/honpas/xzz/siesta-20260520/BiShengCompiler-4.2.0.2-aarch64-linux/bin/clang++
export OMPI_FC=/share/honpas/xzz/siesta-20260520/BiShengCompiler-4.2.0.2-aarch64-linux/bin/flang

export LD_LIBRARY_PATH="${{MPI_HOME}}/lib:${{UCG_HOME}}/lib:${{UCX_HOME}}/lib:${{XPMEM_HOME}}/lib:${{BISHENG_LIB}}:${{LD_LIBRARY_PATH:-}}"
export LIBRARY_PATH="${{MPI_HOME}}/lib:${{UCG_HOME}}/lib:${{UCX_HOME}}/lib:${{XPMEM_HOME}}/lib:${{BISHENG_LIB}}:${{LIBRARY_PATH:-}}"
export PATH="${{MPI_HOME}}/bin:${{UCG_HOME}}/bin:${{UCX_HOME}}/bin:${{XPMEM_HOME}}/bin:${{BISHENG_PATH}}:${{PATH}}"

ulimit -s unlimited
"""


def _render_block_runner(slots, args, procs=None, ntpoly_slice_num=1):
    if procs is None:
        procs = args.procs_per_node
    rank_lines = "\n".join(
        f'  echo "rank {i}=$HOSTNAME_FQDN slots={lo}-{hi}"'
        for i, (lo, hi) in enumerate(slots)
    )
    return f"""#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
source ../honpas_env.sh

# Per-task NTPOLY_SLICE_NUM override (honpas_env.sh defaults it to 1). Larger
# blocks/dimers split the density-matrix purification into more slices to keep
# each slice's working set in HBM. The mpirun -x NTPOLY_SLICE_NUM below forwards
# this overridden value to every rank.
export NTPOLY_SLICE_NUM={ntpoly_slice_num}

HOSTNAME_FQDN="$(hostname -f 2>/dev/null || hostname)"
RANKFILE="$PWD/rankfile.local"
{{
{rank_lines}
}} > "$RANKFILE"

rmmod -f sdma-dae
insmod /usr/lib/modules/5.10.0/kernel/drivers/misc/sdma-dae/sdma_dae.ko share_chns=160 safe_mode=0
sync
echo 3 > /proc/sys/vm/drop_caches

  "${{MPIRUN}}" --allow-run-as-root \
  --prefix "${{MPI_PREFIX}}" \
  -host "${{HOSTNAME_FQDN}}:{procs}" \
  -wdir "$PWD" \
  -np {procs} \
  -x PATH -x LD_LIBRARY_PATH -x OPAL_PREFIX \
  -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES \
  -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS \
  -x NTPOLY_SLICE_NUM -x MEMKIND_HBW_NODES \
  -x VERBS_LOG_LEVEL -x UCX_LOG_LEVEL -x UCX_TLS \
  -x NUM_NUMAS={args.num_numa} -x RANKS_PER_NODE={args.procs_per_node} \
  -x NET_DEVICE_ALL \
  -x UCX_RC_VERBS_ROCE_LOCAL_SUBNET=y -x UCX_UD_VERBS_ROCE_LOCAL_SUBNET=y \
  --rankfile "$RANKFILE" --mca rmaps_rank_file_physical true \
  --mca coll ^ucg --mca pml ucx --mca btl ^vader,tcp,openib,uct,ofi,usnic \
  -x UCX_UD_VERBS_ALLOC=thp,md,mmap,heap \
  -x UCX_RC_VERBS_ALLOC=thp,md,mmap,heap \
  -x UCX_RC_VERBS_TX_MIN_SGE=2 \
  -x UCX_UD_VERBS_TX_MIN_SGE=1 \
  "${{APP}}" input.fdf |& tee siesta.out
"""


def _render_cap_runner(slots, args):
    # Caps are tiny H2 jobs, so run a single-core mpirun (procs=1) instead of
    # the full node. Same node-local flags/env as blocks; running "${APP}"
    # directly fails every cap with no final "Total =" line.
    return _render_block_runner(slots, args, procs=1)


def _render_legacy_head_launch(out: Path, args):
    hosts = args.hosts[: args.num_nodes]
    launch_orted = args.launch_orted or f"{args.pkg_root}/launch_orted.sh"
    return f"""#!/bin/bash
set -euo pipefail

# Legacy path: one head-node mpirun launching all blocks remotely.
# Kept for comparison only. The recommended path is submit_per_node_local.sh.

HOSTS=({" ".join(hosts)})
REMOTE_OUT_DIR="${{REMOTE_OUT_DIR:-{out.resolve()}}}"
SIESTA="${{SIESTA:-siesta}}"
LAUNCH_ORTED="${{LAUNCH_ORTED:-{launch_orted}}}"
export OMP_NUM_THREADS={args.omp_threads}
export OPENBLAS_NUM_THREADS={args.omp_threads}
export MKL_NUM_THREADS={args.omp_threads}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

cd "$REMOTE_OUT_DIR"
for ((i=0;i<{args.num_nodes};i++)); do
  h=${{HOSTS[$i]}}
  blk=$(printf "block_%04d" "$i")
  sed "s/__HOST__/$h/" rankfile_template.txt > "$blk/rankfile.txt"
  (
    cd "$blk"
    mpirun -np {args.num_numa} --host "$h" \
      --rankfile rankfile.txt --bind-to core --report-bindings \
      -x OMP_NUM_THREADS -x MKL_NUM_THREADS -x OPENBLAS_NUM_THREADS \
      -x OMP_PROC_BIND -x OMP_PLACES \
      --mca orte_launch_agent "$LAUNCH_ORTED" \
      "$SIESTA" input.fdf > siesta.out 2>&1
  ) &
done
wait
python3 combine_results.py
"""


def _render_submit_script(out: Path, args):
    hosts = args.hosts[: args.num_nodes]
    ssh_user = f"{args.ssh_user}@" if args.ssh_user else ""
    return f"""#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

HOSTS=({" ".join(hosts)})
REMOTE_OUT_DIR="${{REMOTE_OUT_DIR:-{args.remote_out_dir or out.resolve()}}}"
CAP_HOST="${{CAP_HOST:-${{HOSTS[0]}}}}"
LOG_DIR="${{LOG_DIR:-$PWD/launch_logs}}"
mkdir -p "$LOG_DIR"

for ((i=0;i<{args.num_nodes};i++)); do
  h=${{HOSTS[$i]}}
  blk=$(printf "block_%04d" "$i")
  ssh "{ssh_user}$h" "cd '$REMOTE_OUT_DIR/$blk' && bash ./run_local.sh" \
    > "$LOG_DIR/$blk.$h.log" 2>&1 &
done
wait

# MBE(2) joined dimers: one per cut, run after the blocks free the nodes.
NUM_CUTS=$(( {args.num_nodes} - 1 ))
for ((k=0;k<NUM_CUTS;k++)); do
  h=${{HOSTS[$k % {args.num_nodes}]}}
  dmr=$(printf "dimer_%04d" "$k")
  ssh "{ssh_user}$h" "cd '$REMOTE_OUT_DIR/$dmr' && bash ./run_local.sh" \
    > "$LOG_DIR/$dmr.$h.log" 2>&1 &
done
wait

ssh "{ssh_user}${{CAP_HOST}}" "cd '$REMOTE_OUT_DIR' && for d in cap_*; do (cd \\\"\$d\\\" && bash ./run_local.sh); done" \
  > "$LOG_DIR/caps.$CAP_HOST.log" 2>&1

python3 combine_results.py
"""


def _render_combine_script():
    return """#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import time
from pathlib import Path

TOTAL_RE = re.compile(r"^\\s*siesta:.*Total\\s*=\\s*(-?\\d+\\.\\d+)", re.MULTILINE)


def parse_energy(path: Path):
    if not path.exists():
        return None
    text = path.read_text()
    matches = TOTAL_RE.findall(text)
    return float(matches[-1]) if matches else None


def main():
    _t0 = time.perf_counter()
    root = Path(__file__).resolve().parent
    schedule = json.loads((root / "schedule.json").read_text())
    num_nodes = schedule["num_nodes"]
    num_caps = max(0, num_nodes - 1)
    num_dimers = len(schedule.get("dimers", []))

    block_rows = []
    cap_rows = []
    dimer_rows = []
    missing = []

    for i in range(num_nodes):
        block_dir = root / f"block_{i:04d}"
        energy = parse_energy(block_dir / "siesta.out")
        block_rows.append({"block_id": i, "energy_ev": energy})
        if energy is None:
            missing.append(str(block_dir / "siesta.out"))

    for i in range(num_caps):
        cap_dir = root / f"cap_{i:04d}"
        energy = parse_energy(cap_dir / "siesta.out")
        cap_rows.append({"cap_id": i, "energy_ev": energy})
        if energy is None:
            missing.append(str(cap_dir / "siesta.out"))

    for i in range(num_dimers):
        dimer_dir = root / f"dimer_{i:04d}"
        energy = parse_energy(dimer_dir / "siesta.out")
        dimer_rows.append({"dimer_id": i, "energy_ev": energy})
        if energy is None:
            missing.append(str(dimer_dir / "siesta.out"))

    block_e = [r["energy_ev"] for r in block_rows]
    cap_e = [r["energy_ev"] for r in cap_rows]
    dimer_e = [r["energy_ev"] for r in dimer_rows]
    have_blocks = len(block_e) == num_nodes and all(e is not None for e in block_e)
    have_caps = len(cap_e) == num_caps and all(e is not None for e in cap_e)
    have_dimers = num_dimers > 0 and all(e is not None for e in dimer_e)

    # MFCC(1): Σ E(block) − Σ E(cap). Plain H-cap MFCC carries ~1.24 eV/cut;
    # this is kept only as the uncorrected reference.
    e_mfcc = (sum(block_e) - sum(cap_e)) if (have_blocks and have_caps) else None

    # MBE(2): for each cut k (between block k and block k+1) the joined dimer
    # restores the real bond, giving the increment
    #   Δ_k = E(dimer_k) − E(block_k) − E(block_{k+1}) + E(cap_k)
    # (the +cap_k un-subtracts the cap that no longer exists in the dimer; the
    # cap terms cancel exactly over all cuts). E^(2) = E_mfcc + Σ Δ_k. With no
    # dimers we cannot form MBE(2) and fall back to MFCC(1) — flagged below.
    e_mbe2 = None
    increments = []
    if have_blocks and have_caps and have_dimers and num_dimers == num_caps:
        e_mbe2 = e_mfcc
        for k in range(num_dimers):
            inc = dimer_e[k] - block_e[k] - block_e[k + 1] + cap_e[k]
            increments.append({"cut": k, "increment_ev": inc})
            e_mbe2 += inc

    if e_mbe2 is not None:
        method, e_total = "MBE(2)", e_mbe2
    elif e_mfcc is not None:
        method, e_total = "MFCC(1) (no dimers — run dimer_* jobs to apply MBE(2))", e_mfcc
    else:
        method, e_total = None, None

    summary = {
        "num_nodes": num_nodes,
        "num_caps": num_caps,
        "num_dimers": num_dimers,
        "method": method,
        "E_total_ev": e_total,
        "E_mfcc_ev": e_mfcc,
        "E_mbe2_ev": e_mbe2,
        "mbe2_corrections_ev": increments,
        "missing_outputs": missing,
        "blocks": block_rows,
        "caps": cap_rows,
        "dimers": dimer_rows,
    }
    (root / "weak_scaling_results.json").write_text(json.dumps(summary, indent=2) + "\\n")

    print(f"[combine-time] parse+sum = {time.perf_counter()-_t0:.3f}s  ({num_nodes} blocks, {num_caps} caps, {num_dimers} dimers)")
    print("method =", method)
    print("blocks =", block_rows)
    print("caps   =", cap_rows)
    print("dimers =", dimer_rows)
    print("E_MFCC(1) =", e_mfcc)
    print("E_MBE(2)  =", e_mbe2)
    if increments:
        for inc in increments:
            print(f"  cut {inc['cut']}: +{inc['increment_ev']:.6f} eV")
    print("E_total  =", e_total)
    print("results ->", root / "weak_scaling_results.json")


if __name__ == "__main__":
    main()
"""


def _write_launch_artifacts(out: Path, schedule, args):
    slots = _slot_ranges(args)
    # Record the per-task NTPOLY_SLICE_NUM so a generated run is reproducible /
    # inspectable. Read straight from the CLI; default 1 each.
    block_slice = getattr(args, "block_slice_num", 1)
    dimer_slice = getattr(args, "dimer_slice_num", 1)
    schedule["ntpoly_slice_num"] = {"block": block_slice, "dimer": dimer_slice}
    _write(out / "schedule.json", json.dumps(schedule, indent=2) + "\n")
    _write(out / "honpas_env.sh", _render_env_script(args), 0o755)
    _write(out / "combine_results.py", _render_combine_script(), 0o755)
    _write(out / "submit_per_node_local.sh", _render_submit_script(out, args), 0o755)
    _write(out / "launch_head_mpirun.sh", _render_legacy_head_launch(out, args), 0o755)

    rf = [
        "# Legacy head-node launch rankfile template.",
        "# __HOST__ is substituted by launch_head_mpirun.sh.",
        "# Verify NUMA->CPU mapping with numactl -H or lscpu -e before use.",
    ]
    for i, (lo, hi) in enumerate(slots):
        rf.append(f"rank {i}=__HOST__ slots={lo}-{hi}")
    _write(out / "rankfile_template.txt", "\n".join(rf) + "\n")

    # block_slice / dimer_slice are read from the CLI above (default 1).
    for b in range(schedule["num_nodes"]):
        _write(
            out / f"block_{b:04d}" / "run_local.sh",
            _render_block_runner(slots, args, ntpoly_slice_num=block_slice),
            0o755,
        )
    for c in range(len(schedule["caps"])):
        _write(out / f"cap_{c:04d}" / "run_local.sh", _render_cap_runner(slots, args), 0o755)
    # Joined dimers run like blocks (full-node ntpoly job).
    for k in range(len(schedule.get("dimers", []))):
        _write(
            out / f"dimer_{k:04d}" / "run_local.sh",
            _render_block_runner(slots, args, ntpoly_slice_num=dimer_slice),
            0o755,
        )
    # Unfragmented baseline: launch on one node like a block (ntpoly).
    _write(out / "full" / "run_local.sh", _render_block_runner(slots, args), 0o755)


def main():
    args = _parse_args()
    if args.num_nodes > len(args.hosts):
        raise SystemExit(
            f"num-nodes={args.num_nodes} > available hosts {len(args.hosts)}"
        )
    if not Path(args.gen_script).exists():
        raise SystemExit(
            f"gen script not found: {args.gen_script}"
        )
    for name in ("block_slice_num", "dimer_slice_num"):
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be >= 1, got {getattr(args, name)}")
    if args.omp_threads is None:
        args.omp_threads = args.cores_per_numa - args.skip_cores
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    target_atoms = args.atoms_per_node * args.num_nodes
    _t0 = time.perf_counter()
    mol, n_c = _generate_chain(args)
    _t1 = time.perf_counter()
    real_atoms = 3 * n_c + 2
    print(
        f"target {target_atoms} atoms -> C{n_c}H{2*n_c+2} = {real_atoms} atoms; "
        f"{args.num_nodes} nodes x ~{args.atoms_per_node} atoms/node",
        flush=True,
    )
    print(f"generated {mol.natoms} atoms", flush=True)

    schedule, block_mols, cap_mols, dimer_mols = _build_fragments(mol, n_c, args)
    _t2 = time.perf_counter()
    _write_inputs(out, mol, block_mols, cap_mols, dimer_mols, args)
    _t3 = time.perf_counter()
    _write_launch_artifacts(out, schedule, args)
    _t4 = time.perf_counter()
    print(
        f"[gen-time] chain={_t1-_t0:.2f}s  "
        f"fragments(h_to_c+build)={_t2-_t1:.2f}s  "
        f"write_inputs={_t3-_t2:.2f}s  "
        f"write_artifacts={_t4-_t3:.2f}s  "
        f"total={_t4-_t0:.2f}s",
        flush=True,
    )

    print(f"\n{args.num_nodes} blocks (capped, MFCC-style):", flush=True)
    for block in schedule["blocks"]:
        print(
            f"  block {block['block_id']}: host {block['host']} | "
            f"{block['natoms']} atoms, {block['carbons']} C, "
            f"{block['caps_added']} cap H",
            flush=True,
        )
    print(f"{len(cap_mols)} conjugate caps (H2, serial reference jobs)", flush=True)
    if dimer_mols:
        print(
            f"{len(dimer_mols)} joined dimers (MBE(2), one per cut; ~2x block size):",
            flush=True,
        )
        for d in schedule["dimers"]:
            print(
                f"  dimer {d['dimer_id']}: cut {d['cut']} (blocks "
                f"{d['left_block']}+{d['right_block']}) | host {d['host']} | "
                f"{d['natoms']} atoms, {d['carbons']} C, {d['caps_added']} cap H",
                flush=True,
            )
        print(
            "  combine_results.py applies MBE(2): "
            "E = Σ blocks − Σ caps + Σ_c [E(dimer_c) − E(block) − E(cap)]",
            flush=True,
        )
    else:
        print(
            "MBE order 1: no dimers — combiner uses plain MFCC(1) (~1.24 eV/cut).",
            flush=True,
        )
    print(f"full-chain baseline (unfragmented, ntpoly) -> {out / 'full'}", flush=True)
    print(
        f"NTPOLY_SLICE_NUM        -> block={args.block_slice_num}, "
        f"dimer={args.dimer_slice_num}",
        flush=True,
    )
    print(f"\nschedule                -> {out / 'schedule.json'}")
    print(f"recommended launcher    -> {out / 'submit_per_node_local.sh'}")
    print(f"legacy launcher         -> {out / 'launch_head_mpirun.sh'}")
    print(f"result combiner         -> {out / 'combine_results.py'}")


if __name__ == "__main__":
    main()
