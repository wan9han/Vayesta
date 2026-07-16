#!/usr/bin/env python3
"""Weak-scaling sweep for capped alpha-helix polyglycine (TRS2 + MBE(2)).

Runs the 1/2/4/8/16-node weak-scaling test in one shot, small-to-large, and
collects everything under one folder. Each scale gets its own subfolder with:

  nXX/
    schedule.json              <- generation schedule (json #1, always)
    weak_scaling_results.json  <- MFCC(1)/MBE(2) energies (json #2, from combine)
    block_*/siesta.out  dimer_*/siesta.out  cap_*/siesta.out   <- per-job logs
    launch_logs/  submit_per_node_local.sh  combine_results.py

n=1 is special: 0 cuts -> no dimers/caps, so combine reports MFCC(1) only (the
single block IS the unfragmented chain). It still produces both json.

Execution uses the RECOMMENDED launcher ``submit_per_node_local.sh`` (SSH into
each node, run ``run_local.sh`` there, then combine) -- NOT the legacy head-node
MPI launcher. A ``--executor dev-local`` mode runs each job with plain mpirun on
this machine for testing without intranet nodes.

Usage (intranet, the intended path):
  python3 weak_scale_protein_all.py \\
    --residues-per-node 50 --nodes 1 2 4 8 16 \\
    --big-out /share/.../ws_protein_sweep \\
    --hosts 71.20.27.21 71.20.27.22 ... (16 hosts) \\
    --remote-out-dir /share/.../ws_protein_sweep   # shared path visible to all nodes

Usage (dev smoke test, no intranet):
  python3 weak_scale_protein_all.py --residues-per-node 6 --nodes 1 2 \\
    --big-out /tmp/ws_smoke --executor dev-local
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
GEN = REPO / "weak_scale_protein.py"
# Same 16 intranet node IPs the generator (weak_scale_pe.py) uses by default.
from weak_scale_pe import DEFAULT_HOSTS
# Used only by --executor dev-local (intranet path reads SIESTA from run_local.sh).
DEV_SIESTA = os.environ.get("SIESTA_BIN", "/home/xzz2/huawei-siesta/siesta-install/bin/siesta")


def gen_scale(out_dir: Path, rpn: int, n_nodes: int, hosts, remote_base, extra):
    """Generate one scale via weak_scale_protein.py (ntpoly, no full baseline).

    remote_base: if set, the node-visible top path (head and compute nodes may see
    different paths); the scale subdir name is appended. If None, the scale dir's
    own resolved path is used (correct when --big-out is on a shared /share path)."""
    if remote_base:
        remote = str(Path(remote_base) / out_dir.name)
    else:
        remote = str(out_dir.resolve())
    cmd = [
        sys.executable, str(GEN),
        "--residues-per-node", str(rpn),
        "--num-nodes", str(n_nodes),
        "--out-dir", str(out_dir),
        "--remote-out-dir", remote,
        "--hosts", *hosts[:n_nodes],
        "--no-full-baseline",
        "--solution-method", "ntpoly",
        "--full-solution-method", "ntpoly",
    ]
    cmd += extra
    print(f"\n[gen] n={n_nodes}: {' '.join(cmd[:6])} ...", flush=True)
    subprocess.run(cmd, check=True)


def exec_submit(out_dir: Path):
    """Recommended launcher: SSH into each node, run run_local.sh, then combine."""
    sub = out_dir / "submit_per_node_local.sh"
    if not sub.exists():
        raise FileNotFoundError(f"no submit script in {out_dir} (generation failed?)")
    print(f"[run] submit_per_node_local.sh in {out_dir}", flush=True)
    subprocess.run(["bash", str(sub)], cwd=str(out_dir), check=True)


def exec_dev_local(out_dir: Path, siesta: str, procs: int):
    """Dev fallback: run each job with local mpirun, then combine. Mirrors what
    submit does (blocks, then dimers, then caps) but on one machine."""
    sched = json.loads((out_dir / "schedule.json").read_text())
    n_nodes = sched["num_nodes"]
    n_caps = len(sched.get("caps", []))
    n_dim = len(sched.get("dimers", []))

    def run_job(sub: str, p: int, timeout: float):
        d = out_dir / sub
        if not (d / "input.fdf").exists():
            return
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = "1"
        t0 = time.perf_counter()
        try:
            with open(d / "input.fdf") as fh, open(d / "siesta.out", "w") as so:
                subprocess.run(["mpirun", "-np", str(p), siesta], cwd=str(d),
                               stdin=fh, stdout=so, stderr=subprocess.STDOUT,
                               env=env, timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"    {sub}: TIMEOUT", flush=True)
        print(f"    {sub}: {time.perf_counter()-t0:.0f}s", flush=True)

    print(f"[run:dev-local] {n_nodes} blocks, {n_dim} dimers, {n_caps} caps", flush=True)
    for b in range(n_nodes):
        run_job(f"block_{b:04d}", procs, 900)
    for k in range(n_dim):
        run_job(f"dimer_{k:04d}", procs, 900)
    for c in range(n_caps):
        run_job(f"cap_{c:04d}", 1, 120)          # caps are H2 -> 1 rank
    combine = out_dir / "combine_results.py"
    if combine.exists():
        subprocess.run([sys.executable, str(combine)], cwd=str(out_dir), check=False)


def summarize(big_out: Path):
    """Cross-scale summary: per-residue energy consistency (weak-scaling check)."""
    rows = []
    for sd in sorted(big_out.glob("n*")):
        rj = sd / "weak_scaling_results.json"
        sj = sd / "schedule.json"
        if not (rj.exists() and sj.exists()):
            print(f"  [warn] {sd.name}: missing json (schedule={sj.exists()}, results={rj.exists()})")
            continue
        r = json.loads(rj.read_text())
        s = json.loads(sj.read_text())
        rpn = s["residues_per_node"]
        N = s["num_nodes"]
        e = r.get("E_total_ev")
        rows.append({
            "nodes": N,
            "residues_per_node": rpn,
            "total_residues": rpn * N,
            "method": r.get("method"),
            "E_total_ev": e,
            "E_per_residue_ev": (e / (rpn * N)) if e is not None else None,
            "E_mfcc_ev": r.get("E_mfcc_ev"),
            "E_mbe2_ev": r.get("E_mbe2_ev"),
            "n_blocks": N,
            "n_dimers": len(r.get("dimers", [])),
            "n_caps": len(r.get("caps", [])),
            "missing": r.get("missing_outputs", []),
        })
    (big_out / "weak_scale_summary.json").write_text(json.dumps(rows, indent=2) + "\n")

    print("\n" + "=" * 78)
    print(f"{'nodes':>5} {'res/node':>8} {'total':>6} {'method':>10} "
          f"{'E_total (eV)':>16} {'E/residue':>12}")
    print("-" * 78)
    for r in rows:
        e = f"{r['E_total_ev']:.3f}" if r["E_total_ev"] is not None else "FAIL"
        per = f"{r['E_per_residue_ev']:.5f}" if r["E_per_residue_ev"] is not None else "-"
        print(f"{r['nodes']:>5} {r['residues_per_node']:>8} {r['total_residues']:>6} "
              f"{str(r['method']):>10} {e:>16} {per:>12}")
    print("=" * 78)
    # weak-scaling consistency: per-residue energy should be ~constant across N
    pers = [r["E_per_residue_ev"] for r in rows if r["E_per_residue_ev"] is not None]
    if len(pers) >= 2:
        spread = max(pers) - min(pers)
        print(f"per-residue energy spread across scales: {spread:.5f} eV/residue "
              f"(weak-scaling consistency; should be small)")
    print(f"\nsummary -> {big_out / 'weak_scale_summary.json'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--residues-per-node", type=int, default=50,
                    help="glycine residues per node (fixed across scales = weak scaling)")
    ap.add_argument("--nodes", type=int, nargs="+", default=[1, 2, 4, 8, 16],
                    help="node counts to sweep (default 1 2 4 8 16, small->large)")
    ap.add_argument("--big-out", required=True, help="top-level output folder")
    ap.add_argument("--hosts", nargs="+", default=list(DEFAULT_HOSTS),
                    help="node hostnames/IPs (default: the 16 intranet nodes in "
                         "weak_scale_pe.DEFAULT_HOSTS; need >= max(nodes))")
    ap.add_argument("--remote-out-dir", default=None,
                    help="shared path visible to all nodes (for SSH submit). "
                         "Defaults to big-out's resolved path.")
    ap.add_argument("--executor", choices=["submit", "dev-local"], default="submit",
                    help="submit = recommended SSH launcher (intranet); "
                         "dev-local = local mpirun on this machine (testing)")
    ap.add_argument("--dev-procs", type=int, default=8,
                    help="mpi ranks per job in dev-local mode")
    ap.add_argument("--gen-args", nargs=argparse.REMAINDER, default=[],
                    help="extra args forwarded to weak_scale_protein.py")
    args = ap.parse_args()

    big_out = Path(args.big_out)
    big_out.mkdir(parents=True, exist_ok=True)
    remote_base = args.remote_out_dir  # None, or node-visible top path

    # hosts: default = the 16 intranet nodes (DEFAULT_HOSTS); dev-local can use
    # dummies (no SSH) if the user passes e.g. --hosts localhost ... localhost.
    if len(args.hosts) < max(args.nodes):
        ap.error(f"need >= {max(args.nodes)} hosts for max node count, got {len(args.hosts)}")
    hosts = args.hosts

    t_all = time.perf_counter()
    for N in sorted(args.nodes):
        scale_dir = big_out / f"n{N:02d}"
        t0 = time.perf_counter()
        print(f"\n{'#' * 70}\n# scale N={N} nodes  ->  {scale_dir}\n{'#' * 70}", flush=True)
        gen_scale(scale_dir, args.residues_per_node, N, hosts, remote_base, args.gen_args)
        if args.executor == "submit":
            exec_submit(scale_dir)
        else:
            exec_dev_local(scale_dir, DEV_SIESTA, args.dev_procs)
        # report the two json
        for j in ("schedule.json", "weak_scaling_results.json"):
            p = scale_dir / j
            print(f"  {j}: {'OK' if p.exists() else 'MISSING'}", flush=True)
        print(f"[scale N={N} done in {time.perf_counter()-t0:.0f}s]", flush=True)

    summarize(big_out)
    print(f"\n[all scales done in {time.perf_counter()-t_all:.0f}s]")


if __name__ == "__main__":
    main()
