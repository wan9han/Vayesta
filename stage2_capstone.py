#!/usr/bin/env python3
"""Stage 2 capstone — MFCC energy reproduction + independent-run structure.

On a larger chain (default C16H34) we split into 1 (full), 2, 4, 8 fragments
and report, per configuration:

  * E_full vs E_MFCC and the honest error,
  * per-fragment SIESTA wall time — each fragment is an *independent* run,
    i.e. one fragment per machine is the weak-scaling structure,
  * the trade-off: more fragments -> smaller/faster/parallel runs, but more
    cuts -> larger (linear) MFCC error.

This is the honest end-to-end picture for the project goal: an energy-
reproducing fragmentation whose combination error scales linearly with the
number of cuts while each compute unit stays small and independent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from energy_first.mfcc_geometry import build_mfcc, pick_cuts
from energy_first.molecule import parse_gen_fdf_text
from energy_first.siesta_backend import run_siesta


def _fp(mol):
    arr = np.round(np.asarray(mol.coords), 8)
    return hashlib.sha256(("".join(mol.elements) + "|" + arr.tobytes().hex()).encode()).hexdigest()


def gen(python, gen_script, n):
    p = subprocess.run([python, gen_script, str(n)], capture_output=True, text=True, check=True)
    return parse_gen_fdf_text(p.stdout, label=f"PE_{n}C_full")


def run_timed(mol, cache, root, ctx, label):
    key = _fp(mol)
    if key in cache:
        r = cache[key]
        return r["energy_ev"], 0.0, r["natoms"], r["formula"]
    d = root / f"run_{len(cache):04d}_{label}"
    t0 = time.perf_counter()
    r = run_siesta(mol, d, siesta_bin=ctx["bin"], pseudo_dir=ctx["psd"],
                   label=label, basis_size=ctx["basis"])
    dt = time.perf_counter() - t0
    cache[key] = r
    return r["energy_ev"], dt, r["natoms"], r["formula"]


def config(n_carbons, num_fragments, ctx):
    root = ctx["root"] / f"C{n_carbons}_f{num_fragments}"
    root.mkdir(parents=True, exist_ok=True)
    full = gen(ctx["py"], ctx["gen"], n_carbons)

    e_full, t_full, _, _ = run_timed(full, ctx["cache"], root, ctx, f"C{n_carbons}_full")
    cuts = pick_cuts(full, num_fragments)
    frags, caps = build_mfcc(full, cuts)

    frag_rows = []
    t_frag_total = 0.0
    e_frag_total = 0.0
    for i, f in enumerate(frags):
        e, dt, na, form = run_timed(f, ctx["cache"], root, ctx, f"C{n_carbons}_f{num_fragments}_frag{i}")
        frag_rows.append({"formula": form, "natoms": na, "E_ev": e, "wall_s": round(dt, 3),
                          "max_atoms_in_fragment": na})
        t_frag_total += dt
        e_frag_total += e if e is not None else 0.0
    cap_rows = []
    e_cap_total = 0.0
    for i, c in enumerate(caps):
        e, dt, na, form = run_timed(c, ctx["cache"], root, ctx, f"C{n_carbons}_f{num_fragments}_cap{i}")
        cap_rows.append({"formula": form, "E_ev": e})
        e_cap_total += e if e is not None else 0.0

    ok = all(r["E_ev"] is not None for r in frag_rows + cap_rows) and e_full is not None
    e_mfcc = (e_frag_total - e_cap_total) if ok else None
    max_frag_atoms = max((r["natoms"] for r in frag_rows), default=0)
    max_frag_wall = max((r["wall_s"] for r in frag_rows), default=0.0)
    return {
        "n_carbons": n_carbons, "natoms_full": full.natoms,
        "num_fragments": num_fragments, "num_cuts": len(cuts),
        "basis": ctx["basis"],
        "E_full_ev": e_full, "full_wall_s": round(t_full, 3),
        "E_mfcc_ev": e_mfcc,
        "error_ev": (e_mfcc - e_full) if ok else None,
        "error_per_atom_ev": ((e_mfcc - e_full) / full.natoms) if ok else None,
        "max_fragment_atoms": max_frag_atoms,
        "max_fragment_wall_s": round(max_frag_wall, 3),
        "sum_fragment_wall_s_serial": round(t_frag_total, 3),
        "fragments": frag_rows, "caps": cap_rows, "ok": ok,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--siesta-bin", default="/home/xzz2/huawei-siesta/siesta-install/bin/siesta")
    ap.add_argument("--pseudo-dir", default="/home/xzz2/huawei-siesta/testcases")
    ap.add_argument("--gen-script", default="/home/xzz2/huawei-siesta/testcases/gen.py")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--work-root", default="/tmp/stage2-runs")
    ap.add_argument("--chain", type=int, default=16)
    ap.add_argument("--fragments", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--basis", default="SZ")
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)

    ctx = {"bin": args.siesta_bin, "psd": args.pseudo_dir, "gen": args.gen_script,
           "py": args.python, "root": Path(args.work_root), "cache": {}, "basis": args.basis}

    # num_fragments=1 is the full-system reference (E_full itself).
    configs = [1] + list(args.fragments)
    results = []
    print(f"\n=== Stage 2 capstone: C{args.chain}H{2*args.chain+2}, basis={args.basis} ===")
    print(f"{'frags':>6}{'cuts':>6}{'E_total':>15}{'err(eV)':>12}{'eV/atom':>10}"
          f"{'maxfrag_atoms':>15}{'maxfrag_wall_s':>15}{'serial_sum_s':>13}")
    for nf in configs:
        r = config(args.chain, nf, ctx)
        results.append(r)
        etot = r["E_mfcc_ev"] if nf > 1 else r["E_full_ev"]
        err = r["error_ev"] if nf > 1 else 0.0
        epa = r["error_per_atom_ev"] if nf > 1 else 0.0
        print(f"{nf:>6}{r['num_cuts']:>6}{etot:>15.4f}{err:>12.4f}{epa:>10.5f}"
              f"{r['max_fragment_atoms']:>15}{r['max_fragment_wall_s']:>15}"
              f"{r['sum_fragment_wall_s_serial']:>13}")

    report = {"stage": "stage2_capstone", "chain": args.chain, "basis": args.basis,
              "configs": results,
              "note": "each fragment is an independent SIESTA run (1 frag/machine = weak scaling); "
                      "error grows linearly with cuts, fragment size/wall-time shrink with more frags."}
    out = Path(args.report) if args.report else ctx["root"] / "stage2_capstone_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nfull-system wall = {results[0]['full_wall_s']}s ; report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
