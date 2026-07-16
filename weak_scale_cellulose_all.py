#!/usr/bin/env python3
"""Weak-scaling sweep for cellulose I-beta (TRS2 + MBE(2)).

Runs 1/2/4/8/16-node weak-scaling in one shot, small-to-large, collected under
one folder. Each scale gets its own subfolder with schedule.json +
weak_scaling_results.json + block/dimer/cap logs. Execution uses the recommended
submit_per_node_local.sh (SSH each node, run_local.sh, then combine); --executor
dev-local runs jobs with plain mpirun on this machine for testing.

Cellulose is the clean large-gap (4.2 eV, stable) bio-polymer: TRS2 converges at
any scale. PE-parity per node = 296 glucose (~15990 SZ orbitals ~ PE's 16000).

Usage (intranet, PE parity):
  python3 weak_scale_cellulose_all.py --big-out /share/.../ws_cellu_sweep
  # --glucose-per-node defaults to 296 (PE parity); --hosts defaults to 16 intranet nodes.

Usage (dev smoke test):
  python3 weak_scale_cellulose_all.py --glucose-per-node 4 --nodes 1 2 \\
    --big-out /tmp/ws_cellu_smoke --executor dev-local
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent
GEN = REPO / "weak_scale_cellulose.py"
from weak_scale_pe import DEFAULT_HOSTS
DEV_SIESTA = os.environ.get("SIESTA_BIN", "/home/xzz2/huawei-siesta/siesta-install/bin/siesta")


def gen_scale(out_dir, gpn, n_nodes, hosts, remote_base, extra):
    remote = str(Path(remote_base) / out_dir.name) if remote_base else str(out_dir.resolve())
    cmd = [sys.executable, str(GEN),
           "--glucose-per-node", str(gpn), "--num-nodes", str(n_nodes),
           "--out-dir", str(out_dir), "--remote-out-dir", remote,
           "--hosts", *hosts[:n_nodes], "--no-full-baseline",
           "--solution-method", "ntpoly", "--full-solution-method", "ntpoly"] + extra
    print(f"\n[gen] n={n_nodes} ...", flush=True)
    subprocess.run(cmd, check=True)


def exec_submit(out_dir):
    sub = out_dir / "submit_per_node_local.sh"
    if not sub.exists():
        print(f"[ERROR] submit_per_node_local.sh NOT FOUND in {out_dir}", flush=True)
        print(f"  contents of {out_dir}:", flush=True)
        import os
        for p in sorted(os.listdir(out_dir)):
            print(f"    {p}", flush=True)
        raise FileNotFoundError(f"{sub} -- did gen_scale write it? check weak_scale_cellulose.py output above.")
    print(f"[run] submit_per_node_local.sh in {out_dir}", flush=True)
    subprocess.run(["bash", str(sub)], cwd=str(out_dir), check=True)


def exec_dev_local(out_dir, siesta, procs):
    sched = json.loads((out_dir / "schedule.json").read_text())
    nn, nc, nd = sched["num_nodes"], len(sched.get("caps", [])), len(sched.get("dimers", []))

    def run_job(sub, p, timeout):
        d = out_dir / sub
        if not (d / "input.fdf").exists():
            return
        env = dict(os.environ); env["OMP_NUM_THREADS"] = "1"
        t0 = time.perf_counter()
        try:
            with open(d / "input.fdf") as fh, open(d / "siesta.out", "w") as so:
                subprocess.run(["mpirun", "-np", str(p), siesta], cwd=str(d), stdin=fh,
                               stdout=so, stderr=subprocess.STDOUT, env=env, timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"    {sub}: TIMEOUT", flush=True)
        print(f"    {sub}: {time.perf_counter()-t0:.0f}s", flush=True)

    print(f"[run:dev-local] {nn} blocks, {nd} dimers, {nc} caps", flush=True)
    for b in range(nn):
        run_job(f"block_{b:04d}", procs, 1500)
    for k in range(nd):
        run_job(f"dimer_{k:04d}", procs, 1500)
    for c in range(nc):
        run_job(f"cap_{c:04d}", 1, 120)
    combine = out_dir / "combine_results.py"
    if combine.exists():
        subprocess.run([sys.executable, str(combine)], cwd=str(out_dir), check=False)


def summarize(big_out):
    rows = []
    for sd in sorted(big_out.glob("n*")):
        rj, sj = sd / "weak_scaling_results.json", sd / "schedule.json"
        if not (rj.exists() and sj.exists()):
            print(f"  [warn] {sd.name}: missing json"); continue
        r, s = json.loads(rj.read_text()), json.loads(sj.read_text())
        gpn, N = s["glucose_per_node"], s["num_nodes"]
        e = r.get("E_total_ev")
        rows.append({"nodes": N, "glucose_per_node": gpn, "total_glucose": gpn * N,
                     "total_atoms": s.get("total_chain_atoms"), "method": r.get("method"),
                     "E_total_ev": e,
                     "E_per_glucose_ev": (e / (gpn * N)) if e is not None else None,
                     "E_mfcc_ev": r.get("E_mfcc_ev"), "E_mbe2_ev": r.get("E_mbe2_ev"),
                     "n_dimers": len(r.get("dimers", [])), "missing": r.get("missing_outputs", [])})
    (big_out / "weak_scale_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print("\n" + "=" * 84)
    print(f"{'nodes':>5} {'glu/node':>8} {'atoms':>7} {'method':>8} {'E_total (eV)':>16} {'E/glu':>12}")
    print("-" * 84)
    for r in rows:
        e = f"{r['E_total_ev']:.3f}" if r["E_total_ev"] is not None else "FAIL"
        per = f"{r['E_per_glucose_ev']:.5f}" if r["E_per_glucose_ev"] is not None else "-"
        print(f"{r['nodes']:>5} {r['glucose_per_node']:>8} {str(r['total_atoms']):>7} "
              f"{str(r['method']):>8} {e:>16} {per:>12}")
    print("=" * 84)
    pers = [r["E_per_glucose_ev"] for r in rows if r["E_per_glucose_ev"] is not None]
    if len(pers) >= 2:
        print(f"per-glucose energy spread: {max(pers)-min(pers):.5f} eV/glu (weak-scale consistency)")
    print(f"\nsummary -> {big_out / 'weak_scale_summary.json'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glucose-per-node", type=int, default=296,
                    help="glucose units per node (default 296 = PE parity ~16000 orbitals; "
                         "smaller for dev tests)")
    ap.add_argument("--nodes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--big-out", required=True)
    ap.add_argument("--hosts", nargs="+", default=list(DEFAULT_HOSTS),
                    help="node hostnames/IPs (default 16 intranet nodes; need >= max(nodes))")
    ap.add_argument("--remote-out-dir", default=None,
                    help="shared path visible to all nodes (defaults to --big-out path)")
    ap.add_argument("--executor", choices=["submit", "dev-local"], default="submit")
    ap.add_argument("--dev-procs", type=int, default=8)
    ap.add_argument("--gen-args", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()

    big_out = Path(args.big_out); big_out.mkdir(parents=True, exist_ok=True)
    if len(args.hosts) < max(args.nodes):
        ap.error(f"need >= {max(args.nodes)} hosts, got {len(args.hosts)}")

    t_all = time.perf_counter()
    for N in sorted(args.nodes):
        sd = big_out / f"n{N:02d}"; t0 = time.perf_counter()
        print(f"\n{'#'*68}\n# scale N={N} -> {sd}\n{'#'*68}", flush=True)
        gen_scale(sd, args.glucose_per_node, N, args.hosts, args.remote_out_dir, args.gen_args)
        (exec_submit if args.executor == "submit" else
         lambda d: exec_dev_local(d, DEV_SIESTA, args.dev_procs))(sd)
        for j in ("schedule.json", "weak_scaling_results.json"):
            print(f"  {j}: {'OK' if (sd/j).exists() else 'MISSING'}", flush=True)
        print(f"[N={N} done in {time.perf_counter()-t0:.0f}s]", flush=True)
    summarize(big_out)
    print(f"\n[all scales done in {time.perf_counter()-t_all:.0f}s]")


if __name__ == "__main__":
    main()
