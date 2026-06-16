#!/usr/bin/env python3
"""Stage 3 — concrete weak-scaling evidence: parallel fragment execution.

The weak-scaling claim is that each fragment is an *independent* SIESTA job,
so with one fragment per core/machine the wall time stays bounded as the
fragment (and machine) count grows. This script runs every fragment + cap job
of a chain split two ways and times them:

  * serial   : run jobs one after another  (wall ~= sum of job times, grows
                                             with fragment count)
  * parallel : run jobs concurrently        (wall ~= slowest single job,
                                             ~bounded as fragment count grows)

It then reports E_MFCC (1-body) so the energy result is visible alongside the
timing. This is the concrete, on-this-machine analogue of weak scaling: more
fragments -> more parallel units, parallel wall roughly flat.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from energy_first.mfcc_geometry import build_mfcc, pick_cuts
from energy_first.molecule import parse_gen_fdf_text
from energy_first.siesta_backend import run_siesta


def gen(python, gen_script, n):
    p = subprocess.run([python, gen_script, str(n)], capture_output=True, text=True, check=True)
    return parse_gen_fdf_text(p.stdout, label=f"PE_{n}C_full")


def _job(mol, workdir, ctx, label):
    d = workdir / label
    r = run_siesta(mol, d, siesta_bin=ctx["bin"], pseudo_dir=ctx["psd"],
                   label=label, basis_size=ctx["basis"])
    return label, r["energy_ev"]


def run_config(n_carbons, num_fragments, ctx):
    root = ctx["root"] / f"C{n_carbons}_f{num_fragments}"
    root.mkdir(parents=True, exist_ok=True)
    mol = gen(ctx["py"], ctx["gen"], n_carbons)
    cuts = pick_cuts(mol, num_fragments)
    frags, caps = build_mfcc(mol, cuts)
    jobs = ([(f"frag{i}", f) for i, f in enumerate(frags)]
            + [(f"cap{i}", c) for i, c in enumerate(caps)])

    # serial
    t0 = time.perf_counter()
    serial = {lbl: e for lbl, e in (_job(m, root / "serial", ctx, lbl) for lbl, m in jobs)}
    t_serial = time.perf_counter() - t0

    # parallel (thread pool launches subprocesses concurrently)
    t0 = time.perf_counter()
    parallel = {}
    with ThreadPoolExecutor(max_workers=ctx["workers"]) as ex:
        futs = [ex.submit(_job, m, root / "parallel", ctx, lbl) for lbl, m in jobs]
        for f in futs:
            lbl, e = f.result()
            parallel[lbl] = e
    t_parallel = time.perf_counter() - t0

    e_frags = sum(parallel[f"frag{i}"] for i in range(len(frags)))
    e_caps = sum(parallel[f"cap{i}"] for i in range(len(caps)))
    e_mfcc = e_frags - e_caps
    return {
        "n_carbons": n_carbons, "num_fragments": num_fragments, "num_cuts": len(cuts),
        "num_jobs": len(jobs), "workers": ctx["workers"],
        "serial_wall_s": round(t_serial, 3), "parallel_wall_s": round(t_parallel, 3),
        "speedup": round(t_serial / t_parallel, 2) if t_parallel > 0 else None,
        "E_mfcc_ev": e_mfcc,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--siesta-bin", default="/home/xzz2/huawei-siesta/siesta-install/bin/siesta")
    ap.add_argument("--pseudo-dir", default="/home/xzz2/huawei-siesta/testcases")
    ap.add_argument("--gen-script", default="/home/xzz2/huawei-siesta/testcases/gen.py")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--work-root", default="/tmp/stage3-runs")
    ap.add_argument("--basis", default="SZ")
    ap.add_argument("--workers", type=int, default=min(8, max(2, (os.cpu_count() or 4) // 2)))
    ap.add_argument("--chain", type=int, default=16)
    ap.add_argument("--fragments", type=int, nargs="+", default=[2, 4, 8])
    args = ap.parse_args(argv)

    ctx = {"bin": args.siesta_bin, "psd": args.pseudo_dir, "gen": args.gen_script,
           "py": args.python, "root": Path(args.work_root), "basis": args.basis,
           "workers": args.workers}

    print(f"\n=== Stage 3 weak-scaling demo: C{args.chain}H{2*args.chain+2}, "
          f"basis={args.basis}, workers={args.workers} ===")
    print(f"{'frags':>6}{'cuts':>6}{'jobs':>6}{'serial_wall_s':>15}"
          f"{'parallel_wall_s':>17}{'speedup':>9}{'E_MFCC(eV)':>14}")
    rows = []
    for nf in args.fragments:
        r = run_config(args.chain, nf, ctx)
        rows.append(r)
        print(f"{nf:>6}{r['num_cuts']:>6}{r['num_jobs']:>6}{r['serial_wall_s']:>15}"
              f"{r['parallel_wall_s']:>17}{r['speedup']:>9}{r['E_mfcc_ev']:>14.4f}")
    print("\nAs fragments grow, serial wall grows ~linearly while parallel wall")
    print("stays ~bounded (one SIESTA job per worker) — the weak-scaling structure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
