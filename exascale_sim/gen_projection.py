#!/usr/bin/env python3
"""E级生成时间外推：实测小规模 h_to_c（旧 brute O(N^2) vs 新 cKDTree O(N log N)），
拟合标度后外推到 22,680 节点（8000 原子/节点，n_c≈6.05e7）。

Portable -- no hardcoded paths. Requires numpy + scipy (cKDTree) in the
running python3, and energy_first/ (auto-added via the Vayesta parent dir).
--gen-script is required. Usage:
  python3 gen_projection.py --gen-script /path/to/gen.py
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

# energy_first/ lives in the Vayesta dir (parent of this script's exascale_sim/)
VAYESTA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VAYESTA))

import numpy as np
from scipy.spatial import cKDTree
from energy_first.molecule import parse_gen_fdf_text


def gen_mol(gen_script, n_c):
    # inherit the caller env (numpy/scipy must be on the intranet python3)
    p = subprocess.run(["python3", gen_script, str(n_c)],
                       capture_output=True, text=True, check=True)
    return parse_gen_fdf_text(p.stdout, label=f"C{n_c}")


def setup(mol):
    coords = mol.coords
    cs = [i for i, e in enumerate(mol.elements) if e == "C"]
    cs.sort(key=lambda i: (coords[i, 0], coords[i, 1], coords[i, 2]))
    return coords, coords[cs], cs, [i for i, e in enumerate(mol.elements) if e == "H"]


def brute(coords, cpos, cs, h_idx):
    return {h: cs[int(np.argmin(np.linalg.norm(cpos - coords[h], axis=1)))] for h in h_idx}


def kdtree(coords, cpos, cs, h_idx):
    if not h_idx:
        return {}
    tree = cKDTree(cpos)
    _, nn = tree.query(coords[h_idx], k=1)
    nn = np.atleast_1d(nn)
    return {h: cs[int(j)] for h, j in zip(h_idx, nn)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-script", required=True, help="path to gen.py")
    ap.add_argument("--nodes", type=int, default=22680,
                    help="node count to extrapolate to (default 22680)")
    args = ap.parse_args()

    print("=== measured: h_to_c (old brute / new cKDTree) ===")
    print(f"{'n_c':>8} {'atoms':>8} {'brute(s)':>10} {'kdtree(s)':>10}")
    brute_n, brute_t, kd_n, kd_t = [], [], [], []
    for n_c, do_brute in [(2000, True), (4000, True), (8000, True), (16000, True),
                          (32000, False), (100000, False)]:
        mol = gen_mol(args.gen_script, n_c)
        coords, cpos, cs, h_idx = setup(mol)
        tb = float('nan')
        if do_brute:
            t0 = time.perf_counter(); brute(coords, cpos, cs, h_idx); tb = time.perf_counter() - t0
            brute_n.append(n_c); brute_t.append(tb)
        t0 = time.perf_counter(); kdtree(coords, cpos, cs, h_idx); tk = time.perf_counter() - t0
        kd_n.append(n_c); kd_t.append(tk)
        print(f"{n_c:>8} {mol.natoms:>8} {tb:>10.3f} {tk:>10.4f}")

    a = np.mean([t / n**2 for n, t in zip(brute_n, brute_t)])
    b = np.mean([t / (n * np.log2(n)) for n, t in zip(kd_n, kd_t)])
    NC_E = round((args.nodes * 8000 - 2) / 3)
    t_brute_E = a * NC_E**2
    t_kd_E = b * NC_E * np.log2(NC_E)
    print(f"\n=== extrapolated to {args.nodes} nodes (n_c={NC_E:,}, {3*NC_E+2:,} atoms) ===")
    print(f"  h_to_c old (brute, O(N^2)):    {t_brute_E:,.0f} s = {t_brute_E/86400:,.1f} days")
    print(f"  h_to_c new (cKDTree, O(NlogN)): {t_kd_E:,.1f} s = {t_kd_E/60:.1f} min")
    print(f"  speedup: {t_brute_E/t_kd_E:,.0f} x")


if __name__ == "__main__":
    main()
