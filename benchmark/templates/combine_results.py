#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import time
from pathlib import Path

TOTAL_RE = re.compile(r"^\s*siesta:.*Total\s*=\s*(-?\d+\.\d+)", re.MULTILINE)


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
    (root / "weak_scaling_results.json").write_text(json.dumps(summary, indent=2) + "\n")

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
