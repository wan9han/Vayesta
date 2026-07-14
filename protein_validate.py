#!/usr/bin/env python3
"""Polyglycine MFCC+MBE(2) validation — demonstrate generality beyond PE.

Cuts peptide bonds (C'-N), caps with H, runs SIESTA, compares
E_full vs E_mfcc(1) vs E_mbe(2). The energy-combination math is identical
to PE; only the cut bond type (C'-N vs C-C) and geometry differ.
"""
from __future__ import annotations
import argparse, hashlib, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from energy_first.peptide_chain import generate_polyglycine
from energy_first.molecule import Molecule
from energy_first.siesta_backend import run_siesta

SIESTA = "/home/xzz2/huawei-siesta/siesta-install/bin/siesta"
PSEUDO = "/home/xzz2/huawei-siesta/testcases"


def _dist(a, b):
    return np.linalg.norm(np.asarray(a) - np.asarray(b))


def find_peptide_bonds(mol):
    """C'(carbonyl, bonded to O)-N pairs at ~1.33 Å."""
    co, el = mol.coords, mol.elements
    cs = [i for i, e in enumerate(el) if e == "C"]
    ns = [i for i, e in enumerate(el) if e == "N"]
    os_ = [i for i, e in enumerate(el) if e == "O"]
    bonds = []
    for ci in cs:
        if not any(_dist(co[ci], co[oi]) < 1.3 for oi in os_):
            continue
        for ni in ns:
            if 1.2 < _dist(co[ci], co[ni]) < 1.45:
                bonds.append((ci, ni))
    return bonds


def _connected_components(natoms, coords, cut_set, cutoff=1.7):
    """Union-find on atom pairs < cutoff, excluding cut_set edges."""
    par = list(range(natoms))
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    def union(x, y): par[find(x)] = find(y)
    for i in range(natoms):
        for j in range(i + 1, natoms):
            if frozenset((i, j)) in cut_set: continue
            if _dist(coords[i], coords[j]) < cutoff: union(i, j)
    comp = [find(i) for i in range(natoms)]
    return comp


def _cap(a_coord, b_coord, ch=1.10):
    v = np.asarray(b_coord) - np.asarray(a_coord)
    return np.asarray(a_coord) + v / np.linalg.norm(v) * ch


def fragment(mol, cuts):
    """Split mol at cuts, add H caps. Returns (frag_mols, cap_mols)."""
    co = mol.coords
    cut_set = {frozenset(c) for c in cuts}
    comp = _connected_components(mol.natoms, co, cut_set)
    groups = defaultdict(list)
    for i in range(mol.natoms): groups[comp[i]].append(i)
    frags = sorted(groups.values(), key=lambda g: min(g))

    frag_mols, cap_mols = [], []
    for fi, atoms in enumerate(frags):
        els = [mol.elements[a] for a in atoms]
        m = Molecule(list(els), co[atoms].copy(), label=f"frag_{fi}")
        for (ci, ni) in cuts:
            if ci in atoms: m.append("H", _cap(co[ci], co[ni]))
            if ni in atoms: m.append("H", _cap(co[ni], co[ci]))
        frag_mols.append(m)
    for k, (ci, ni) in enumerate(cuts):
        h1, h2 = _cap(co[ci], co[ni]), _cap(co[ni], co[ci])
        cap_mols.append(Molecule(["H", "H"], np.vstack([h1, h2]), label=f"cap_{k}"))
    return frag_mols, cap_mols, comp


def build_dimer(mol, all_cuts, keep_cut):
    """Join the two fragments sharing keep_cut, cap at other cuts."""
    co = mol.coords
    cut_set = {frozenset(c) for c in all_cuts} - {frozenset(keep_cut)}
    comp = _connected_components(mol.natoms, co, cut_set)
    target = comp[keep_cut[0]]
    atoms = sorted(i for i in range(mol.natoms) if comp[i] == target)
    els = [mol.elements[a] for a in atoms]
    m = Molecule(list(els), co[atoms].copy(), label="dimer")
    for (cj, nj) in all_cuts:
        if (cj, nj) == keep_cut: continue
        if cj in atoms: m.append("H", _cap(co[cj], co[nj]))
        if nj in atoms: m.append("H", _cap(co[nj], co[cj]))
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-residues", type=int, default=6)
    ap.add_argument("--cuts", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--work-root", default="/tmp/protein-validate")
    args = ap.parse_args(argv)

    root = Path(args.work_root); root.mkdir(parents=True, exist_ok=True)
    cache = {}
    def run(mol, label):
        key = hashlib.sha256((label + str(np.round(mol.coords, 8).tobytes())).encode()).hexdigest()
        if key in cache: return cache[key]
        m = Molecule(list(mol.elements), mol.coords.copy(), label)
        r = run_siesta(m, root / label, siesta_bin=SIESTA, pseudo_dir=PSEUDO,
                       label=label, basis_size="SZ", solution_method="diagonali", timeout=300)
        e = r["energy_ev"]
        if e is None: print(f"  ⚠ {label}: FAIL rc={r['returncode']}")
        cache[key] = e; return e

    elements, coords = generate_polyglycine(args.n_residues)
    mol = Molecule(list(elements), coords, label="gly")
    bonds = find_peptide_bonds(mol)
    print(f"polyglycine({args.n_residues}): {mol.natoms} atoms (C/H/N/O="
          f"{int((elements=='C').sum())}/{int((elements=='H').sum())}/"
          f"{int((elements=='N').sum())}/{int((elements=='O').sum())}), "
          f"{len(bonds)} peptide bonds")

    e_full = run(mol, "full")
    print(f"E_full = {e_full:.4f} eV\n" if e_full else "E_full FAILED\n")

    for nc in args.cuts:
        if nc > len(bonds): continue
        step = max(1, len(bonds) // (nc + 1))
        cuts = [bonds[k * step] for k in range(nc)]
        print(f"=== {nc} cut(s) ===")

        frags, caps, comp = fragment(mol, cuts)
        e_frags = [run(f, f"f{nc}_{i}") for i, f in enumerate(frags)]
        e_caps  = [run(c, f"c{nc}_{i}") for i, c in enumerate(caps)]
        e_dimers = []
        for k, cut in enumerate(cuts):
            d = build_dimer(mol, cuts, cut)
            e_dimers.append(run(d, f"d{nc}_{k}"))

        e_mfcc = sum(e_frags) - sum(e_caps)
        err1 = e_mfcc - e_full

        # MBE(2): dimer increments
        frag_idx = {c: i for i, c in enumerate(sorted(set(comp)))}
        e_mbe2 = e_mfcc
        incs = []
        for k, (ci, ni) in enumerate(cuts):
            fi, fj = frag_idx[comp[ci]], frag_idx[comp[ni]]
            inc = e_dimers[k] - e_frags[fi] - e_frags[fj] + e_caps[k]
            incs.append(inc); e_mbe2 += inc
        err2 = e_mbe2 - e_full

        print(f"  MFCC(1): {e_mfcc:.4f}  err={err1:+.4f} eV ({err1/nc:+.4f}/cut)")
        print(f"  MBE(2) : {e_mbe2:.4f}  err={err2:+.4f} eV ({err2/nc:+.4f}/cut)")
        if abs(err2) > 1e-6:
            print(f"  improve: {abs(err1)/abs(err2):.0f}x")
        for k, inc in enumerate(incs):
            print(f"    cut {k}: Δ={inc:+.4f}")
        print()

    print(f"({len(cache)} SIESTA jobs)")


if __name__ == "__main__":
    main()
