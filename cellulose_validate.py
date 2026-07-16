#!/usr/bin/env python3
"""Cellulose I-beta MFCC+MBE(2) validation (TRS2).

Geometry from energy_first/build_cellulose_ibeta.py (clean Nishiyama 2_1 helix,
no tiling strain; 4.2 eV gap, stable with length -> TRS2 converges). MFCC cuts
the C-O GLYCOSIDIC bonds (the beta-1,4 inter-glucose linkage; NOT ring C-O5 or
C-OH), H-caps the cuts, MBE(2) joins dimers. Energy math identical to PE/PEG.
"""
from __future__ import annotations
import argparse, hashlib, sys
from collections import defaultdict, deque
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from energy_first.molecule import Molecule
from energy_first.siesta_backend import run_siesta
from energy_first.build_cellulose_ibeta import build_chain

_REPO = Path(__file__).resolve().parent


def _cap(a_coord, b_coord, ch=1.10):
    v = np.asarray(b_coord) - np.asarray(a_coord)
    return np.asarray(a_coord) + v / np.linalg.norm(v) * ch


def _adjacency(co, cutoff=1.7):
    from scipy.spatial import cKDTree
    nat = len(co)
    tree = cKDTree(co)
    adj = defaultdict(list)
    for i, j in tree.query_pairs(cutoff, output_type="ndarray"):
        i, j = int(i), int(j)
        adj[i].append(j); adj[j].append(i)
    return adj


def find_glycosidic_bonds(mol):
    """C-O glycosidic bonds (beta-1,4 inter-glucose linkage): an O bonded to 2
    C's whose connection survives removing the O = SAME glucose (ring O5);
    whose connection breaks = DIFFERENT glucoses (glycosidic). Returns
    [(C4_idx, O_idx), ...] along the chain."""
    co, el = mol.coords, np.asarray(mol.elements)
    adj = _adjacency(co)
    bonds = []
    for i, e in enumerate(el):
        if e != "O":
            continue
        cs = [j for j in adj[i] if el[j] == "C"]
        if len(cs) != 2:
            continue
        a, b = cs
        # is b reachable from a without going through O i?  (BFS excluding i)
        vis = {i, a}; dq = deque([a]); reach = False
        while dq:
            x = dq.popleft()
            if x == b:
                reach = True; break
            for y in adj[x]:
                if y not in vis:
                    vis.add(y); dq.append(y)
        if not reach:                          # different glucoses -> glycosidic
            bonds.append((min(a, b), i))       # cut (C4, O)
    return bonds


def _heavy_components(mol, cut_set, cutoff=1.7):
    from scipy.spatial import cKDTree
    co = np.asarray(mol.coords); el = np.asarray(mol.elements)
    heavy = [i for i, e in enumerate(el) if e != "H"]
    hidx = {h: k for k, h in enumerate(heavy)}
    pairs = cKDTree(co[heavy]).query_pairs(cutoff, output_type="ndarray")
    par = list(range(len(heavy)))
    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    def union(x, y): par[find(x)] = find(y)
    for ij in pairs:
        li, lj = int(ij[0]), int(ij[1])           # local heavy indices
        gi, gj = heavy[li], heavy[lj]              # global atom indices (cut_set is global)
        if frozenset((gi, gj)) in cut_set:
            continue
        union(li, lj)
    comp_heavy = [find(k) for k in range(len(heavy))]
    comp = [0] * mol.natoms
    for k, h in enumerate(heavy):
        comp[h] = comp_heavy[k]
    Hs = [i for i, e in enumerate(el) if e == "H"]
    if Hs:
        _, nn = cKDTree(co[heavy]).query(co[Hs], k=1)
        for hi, n in zip(Hs, nn):
            comp[hi] = comp_heavy[n]
    return comp


def fragment_cellulose(mol, cuts):
    co = mol.coords
    cut_set = {frozenset(c) for c in cuts}
    comp = _heavy_components(mol, cut_set)
    groups = defaultdict(list)
    for i in range(mol.natoms):
        groups[comp[i]].append(i)
    frags = sorted(groups.values(), key=lambda g: min(g))
    frag_mols = []
    for fi, atoms in enumerate(frags):
        m = Molecule([mol.elements[a] for a in atoms], co[atoms].copy(), label=f"frag_{fi}")
        for (ci, oi) in cuts:
            if ci in atoms:
                m.append("H", _cap(co[ci], co[oi]))
            if oi in atoms:
                m.append("H", _cap(co[oi], co[ci]))
        frag_mols.append(m)
    cap_mols = []
    for k, (ci, oi) in enumerate(cuts):
        h1, h2 = _cap(co[ci], co[oi]), _cap(co[oi], co[ci])
        cap_mols.append(Molecule(["H", "H"], np.vstack([h1, h2]), label=f"cap_{k}"))
    return frag_mols, cap_mols, comp


def build_dimer_cellulose(mol, all_cuts, keep_cut):
    co = mol.coords
    other = [c for c in all_cuts if frozenset(c) != frozenset(keep_cut)]
    comp = _heavy_components(mol, {frozenset(c) for c in other})
    target = comp[keep_cut[0]]
    atoms = sorted(i for i in range(mol.natoms) if comp[i] == target)
    m = Molecule([mol.elements[a] for a in atoms], co[atoms].copy(), label="dimer")
    for (cj, nj) in all_cuts:
        if frozenset((cj, nj)) == frozenset(keep_cut):
            continue
        if cj in atoms:
            m.append("H", _cap(co[cj], co[nj]))
        if nj in atoms:
            m.append("H", _cap(co[nj], co[cj]))
    return m


def _mol_from_chain(n_glucose):
    atoms, bonds = build_chain(n_glucose)
    el = [a.element for a in atoms]
    co = np.array([a.xyz for a in atoms])
    return Molecule(el, co, f"cellu{n_glucose}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glucose", type=int, default=16, help="number of glucose units")
    ap.add_argument("--cuts", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--work-root", default="/tmp/cellulose-validate")
    ap.add_argument("--siesta-bin",
                    default="/share/honpas/xzz/siesta-20260520/siesta/build-clang/Src/siesta")
    ap.add_argument("--pseudo-dir", default=str(_REPO / "pseudos"))
    ap.add_argument("--solution-method", default="ntpoly", choices=["ntpoly", "diagonali"],
                    help="TRS2 (ntpoly) by default -- cellulose's 4.2 eV gap converges cleanly.")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args(argv)

    root = Path(args.work_root); root.mkdir(parents=True, exist_ok=True)
    cache = {}

    def run(mol, label):
        key = hashlib.sha256((label + str(np.round(mol.coords, 8).tobytes())).encode()).hexdigest()
        if key in cache:
            return cache[key]
        m = Molecule(list(mol.elements), mol.coords.copy(), label)
        solver = "diagonali" if label.startswith("c") else args.solution_method
        r = run_siesta(m, root / label, siesta_bin=args.siesta_bin, pseudo_dir=args.pseudo_dir,
                       label=label, basis_size="SZ", solution_method=solver, timeout=args.timeout)
        e = r["energy_ev"]
        if e is None:
            print(f"  ⚠ {label}: FAIL rc={r['returncode']}")
        cache[key] = e
        return e

    mol = _mol_from_chain(args.glucose)
    bonds = find_glycosidic_bonds(mol)
    el = np.asarray(mol.elements)
    print(f"cellulose Ibeta({args.glucose} glu): {mol.natoms} atoms (C/H/O="
          f"{int((el=='C').sum())}/{int((el=='H').sum())}/{int((el=='O').sum())}), "
          f"{len(bonds)} glycosidic bonds")
    e_full = run(mol, "full")
    print(f"E_full = {e_full:.4f} eV\n" if e_full else "E_full FAILED\n")

    for nc in args.cuts:
        if nc > len(bonds):
            continue
        cuts = [bonds[(k * len(bonds)) // (nc + 1)] for k in range(1, nc + 1)]
        print(f"=== {nc} cut(s) ===")
        frags, caps, comp = fragment_cellulose(mol, cuts)
        e_frags = [run(f, f"f{nc}_{i}") for i, f in enumerate(frags)]
        e_caps = [run(c, f"c{nc}_{i}") for i, c in enumerate(caps)]
        e_dimers = [run(build_dimer_cellulose(mol, cuts, c), f"d{nc}_{k}") for k, c in enumerate(cuts)]
        e_mfcc = sum(e_frags) - sum(e_caps)
        err1 = e_mfcc - e_full
        fidx = {c: i for i, c in enumerate(sorted(set(comp)))}
        e_mbe2 = e_mfcc
        for k, (ci, oi) in enumerate(cuts):
            inc = e_dimers[k] - e_frags[fidx[comp[ci]]] - e_frags[fidx[comp[oi]]] + e_caps[k]
            e_mbe2 += inc
            print(f"    cut {k}: Δ={inc:+.4f}")
        err2 = e_mbe2 - e_full
        print(f"  MFCC(1): {e_mfcc:.4f}  err={err1:+.4f} ({err1/nc:+.4f}/cut)")
        print(f"  MBE(2) : {e_mbe2:.4f}  err={err2:+.4f} ({err2/nc:+.4f}/cut)")
        print()
    print(f"({len(cache)} SIESTA jobs)")


if __name__ == "__main__":
    main()
