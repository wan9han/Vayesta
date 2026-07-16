#!/usr/bin/env python3
"""Build finite, saturated cellulose I-beta single chains from Nishiyama et al. crystal coordinates.

Only NumPy is required to generate XYZ/PDB. SciPy is used, when available, for
fast non-bonded contact checking.

The crystallographic source is the neutron/X-ray supplementary CIF for:
Y. Nishiyama, P. Langan, H. Chanzy, JACS 124 (2002) 9074-9082,
DOI 10.1021/ja0257319.

The default 'origin' chain uses the less-disordered dominant hydroxyl-H sites.
The chain is generated exactly with the crystallographic 2_1 screw operation:
    (x,y,z) -> (-x,-y,z+1/2)
and a repeat translation of c = 10.380 A per cell (two glucose units).
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

A = 7.784
B = 8.201
C = 10.380
GAMMA_DEG = 96.55
OH_BOND = 0.980

# Atom order is chemically meaningful and retained in all outputs.
# D atoms from the neutron structure are emitted as ordinary H.
TEMPLATES: Dict[str, List[Tuple[str, str, Tuple[float, float, float]]]] = {
    "origin": [
        ("C1", "C", (0.0138, -0.0415, 0.0433)),
        ("H1", "H", (0.1382, -0.0188, 0.0598)),
        ("C2", "C", (-0.0260, -0.1843, -0.0516)),
        ("H2", "H", (-0.1508, -0.2177, -0.0546)),
        ("C3", "C", (0.0403, -0.1369, -0.1848)),
        ("H3", "H", (0.1667, -0.1323, -0.1840)),
        ("C4", "C", (-0.0071, 0.0304, -0.2250)),
        ("H4", "H", (-0.1316, 0.0176, -0.2425)),
        ("C5", "C", (0.0258, 0.1594, -0.1232)),
        ("H5", "H", (0.1507, 0.1821, -0.1090)),
        ("C6", "C", (-0.0467, 0.3187, -0.1529)),
        ("H6A", "H", (-0.1672, 0.2959, -0.1778)),
        ("H6B", "H", (-0.0407, 0.3872, -0.0765)),
        ("O2", "O", (0.0609, -0.3144, -0.0003)),
        ("HO2", "H", (0.0366, -0.3194, 0.0920)),
        ("O3", "O", (-0.0268, -0.2607, -0.2759)),
        ("HO3", "H", (0.0475, -0.2400, -0.3517)),
        ("O4", "O", (0.0786, 0.0865, -0.3424)),
        ("O5", "O", (-0.0556, 0.0991, -0.0053)),
        ("O6", "O", (0.0480, 0.4030, -0.2540)),
        ("HO6", "H", (-0.0144, 0.4924, -0.2850)),  # dominant occupancy 0.85
    ],
    "center": [
        ("C1", "C", (0.5329, 0.4548, 0.3043)),
        ("H1", "H", (0.6577, 0.4616, 0.3200)),
        ("C2", "C", (0.4745, 0.3178, 0.2094)),
        ("H2", "H", (0.3480, 0.3058, 0.2052)),
        ("C3", "C", (0.5449, 0.3626, 0.0765)),
        ("H3", "H", (0.6716, 0.3753, 0.0793)),
        ("C4", "C", (0.4823, 0.5263, 0.0375)),
        ("H4", "H", (0.3561, 0.5139, 0.0289)),
        ("C5", "C", (0.5409, 0.6574, 0.1388)),
        ("H5", "H", (0.6670, 0.6830, 0.1370)),
        ("C6", "C", (0.4523, 0.8151, 0.1125)),
        ("H6A", "H", (0.3341, 0.7843, 0.0847)),
        ("H6B", "H", (0.4491, 0.8778, 0.1916)),
        ("O2", "O", (0.5277, 0.1664, 0.2520)),
        ("HO2", "H", (0.5035, 0.1593, 0.3448)),  # dominant occupancy 0.57
        ("O3", "O", (0.4865, 0.2326, -0.0081)),
        ("HO3", "H", (0.5104, 0.2733, -0.0961)),
        ("O4", "O", (0.5625, 0.5859, -0.0806)),
        ("O5", "O", (0.4850, 0.6070, 0.2639)),
        ("O6", "O", (0.5420, 0.9137, 0.0169)),
        ("HO6", "H", (0.5184, 1.0275, 0.0326)),  # dominant occupancy 0.68
    ],
}

LOCAL_BONDS = [
    ("C1", "H1"), ("C1", "C2"), ("C1", "O5"),
    ("C2", "H2"), ("C2", "C3"), ("C2", "O2"), ("O2", "HO2"),
    ("C3", "H3"), ("C3", "C4"), ("C3", "O3"), ("O3", "HO3"),
    ("C4", "H4"), ("C4", "C5"), ("C4", "O4"),
    ("C5", "H5"), ("C5", "C6"), ("C5", "O5"),
    ("C6", "H6A"), ("C6", "H6B"), ("C6", "O6"), ("O6", "HO6"),
]

@dataclass
class Atom:
    element: str
    name: str
    residue: int
    xyz: np.ndarray


def lattice_matrix() -> np.ndarray:
    gamma = math.radians(GAMMA_DEG)
    # Fractional row vector f is converted by f @ L.
    return np.array([
        [A, 0.0, 0.0],
        [B * math.cos(gamma), B * math.sin(gamma), 0.0],
        [0.0, 0.0, C],
    ], dtype=float)


def frac_for_residue(frac: Sequence[float], residue_index: int, chain: str) -> np.ndarray:
    """Apply the exact crystallographic screw and unwrap one isolated chain."""
    x, y, z = map(float, frac)
    q, parity = divmod(residue_index, 2)
    if parity == 0:
        return np.array([x, y, z + q], dtype=float)
    # P 1 1 21 operation: -x,-y,z+1/2. Center chain needs +a,+b unwrapping.
    xy_shift = 1.0 if chain == "center" else 0.0
    return np.array([-x + xy_shift, -y + xy_shift, z + 0.5 + q], dtype=float)


def virtual_xyz(chain: str, residue_index: int, atom_name: str) -> np.ndarray:
    L = lattice_matrix()
    for name, _element, frac in TEMPLATES[chain]:
        if name == atom_name:
            return frac_for_residue(frac, residue_index, chain) @ L
    raise KeyError(atom_name)


def build_chain(n: int, chain: str = "origin", endcaps: str = "oh", center: bool = True):
    if n < 1:
        raise ValueError("N must be at least 1")
    if chain not in TEMPLATES:
        raise ValueError(f"Unknown chain {chain!r}")
    if endcaps not in {"oh", "h"}:
        raise ValueError("endcaps must be 'oh' or 'h'")

    L = lattice_matrix()
    atoms: List[Atom] = []
    index: Dict[Tuple[int, str], int] = {}
    bonds: List[Tuple[int, int]] = []

    for r in range(n):
        for name, element, frac in TEMPLATES[chain]:
            xyz = frac_for_residue(frac, r, chain) @ L
            index[(r, name)] = len(atoms)
            atoms.append(Atom(element, name, r + 1, xyz))

    for r in range(n):
        for a, b in LOCAL_BONDS:
            bonds.append((index[(r, a)], index[(r, b)]))
        if r >= 1:
            # O4(r) is between C4(r) and C1(r-1) in this crystallographic orientation.
            bonds.append((index[(r, "O4")], index[(r - 1, "C1")]))

    # Nonreducing C4 end: replace missing C1(-1) by H, preserving C4-O4-X angle.
    o4_idx = index[(0, "O4")]
    missing_c1 = virtual_xyz(chain, -1, "C1")
    v = missing_c1 - atoms[o4_idx].xyz
    h_xyz = atoms[o4_idx].xyz + OH_BOND * v / np.linalg.norm(v)
    h_o4_idx = len(atoms)
    atoms.append(Atom("H", "HO4", 1, h_xyz))
    bonds.append((o4_idx, h_o4_idx))

    # Reducing/anomeric end: use the missing periodic O4(N) direction exactly.
    c1_idx = index[(n - 1, "C1")]
    missing_o4 = virtual_xyz(chain, n, "O4")
    if endcaps == "oh":
        o1_idx = len(atoms)
        atoms.append(Atom("O", "O1T", n, missing_o4.copy()))
        bonds.append((c1_idx, o1_idx))
        missing_c4 = virtual_xyz(chain, n, "C4")
        v = missing_c4 - missing_o4
        h_xyz = missing_o4 + OH_BOND * v / np.linalg.norm(v)
        h_idx = len(atoms)
        atoms.append(Atom("H", "HO1", n, h_xyz))
        bonds.append((o1_idx, h_idx))
    else:
        v = missing_o4 - atoms[c1_idx].xyz
        h_xyz = atoms[c1_idx].xyz + 1.090 * v / np.linalg.norm(v)
        h_idx = len(atoms)
        atoms.append(Atom("H", "H1T", n, h_xyz))
        bonds.append((c1_idx, h_idx))

    if center:
        coords = np.array([a.xyz for a in atoms])
        # Center on the heavy-atom centroid; preserves exact internal geometry.
        heavy = np.array([a.element != "H" for a in atoms])
        shift = coords[heavy].mean(axis=0)
        for atom in atoms:
            atom.xyz = atom.xyz - shift

    return atoms, bonds


def formula(atoms: Sequence[Atom]) -> Dict[str, int]:
    out = {"C": 0, "H": 0, "O": 0}
    for atom in atoms:
        out[atom.element] = out.get(atom.element, 0) + 1
    return out


def formula_string(counts: Dict[str, int]) -> str:
    return "".join(f"{e}{counts.get(e, 0)}" for e in ("C", "H", "O"))


def write_xyz(path: Path, atoms: Sequence[Atom], n: int, chain: str, endcaps: str):
    f = formula_string(formula(atoms))
    with path.open("w", encoding="utf-8") as out:
        out.write(f"{len(atoms)}\n")
        out.write(
            f"cellulose Ibeta; N={n}; chain={chain}; endcaps={endcaps}; formula={f}; "
            f"source=Nishiyama_JACS_2002; axis=z; units=angstrom\n"
        )
        for atom in atoms:
            x, y, z = atom.xyz
            out.write(f"{atom.element:2s} {x:16.9f} {y:16.9f} {z:16.9f}\n")


def write_pdb(path: Path, atoms: Sequence[Atom], bonds: Sequence[Tuple[int, int]], n: int, chain: str, endcaps: str):
    adjacency: Dict[int, List[int]] = {i: [] for i in range(len(atoms))}
    for i, j in bonds:
        adjacency[i].append(j)
        adjacency[j].append(i)
    with path.open("w", encoding="utf-8") as out:
        out.write("HEADER    CELLULOSE I-BETA SINGLE CHAIN\n")
        out.write(f"REMARK    N={n} CHAIN={chain} ENDCAPS={endcaps} SOURCE=NISHIYAMA JACS 2002\n")
        for i, atom in enumerate(atoms, start=1):
            x, y, z = atom.xyz
            atom_name = atom.name[:4].rjust(4)
            out.write(
                f"HETATM{i:5d} {atom_name} CEL A{atom.residue:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {atom.element:>2s}\n"
            )
        for i in range(len(atoms)):
            nbrs = sorted(adjacency[i])
            for k in range(0, len(nbrs), 4):
                out.write(f"CONECT{i+1:5d}" + "".join(f"{j+1:5d}" for j in nbrs[k:k+4]) + "\n")
        out.write("END\n")


def validate(atoms: Sequence[Atom], bonds: Sequence[Tuple[int, int]]) -> Dict[str, object]:
    coords = np.array([a.xyz for a in atoms], dtype=float)
    elements = [a.element for a in atoms]
    bond_set = {tuple(sorted((i, j))) for i, j in bonds}
    bond_lengths = np.array([np.linalg.norm(coords[i] - coords[j]) for i, j in bonds])

    # Find all pairs within 2.2 A, then exclude explicitly bonded pairs.
    close_pairs: Iterable[Tuple[int, int]]
    try:
        from scipy.spatial import cKDTree
        close_pairs = cKDTree(coords).query_pairs(r=2.2, output_type="set")
    except Exception:
        if len(atoms) > 2500:
            close_pairs = []
        else:
            pairs = []
            for i in range(len(atoms)):
                d = np.linalg.norm(coords[i+1:] - coords[i], axis=1)
                pairs.extend((i, i + 1 + int(j)) for j in np.where(d < 2.2)[0])
            close_pairs = pairs

    min_nonbonded = float("inf")
    min_pair = None
    contacts_under_09 = []
    for i, j in close_pairs:
        if tuple(sorted((i, j))) in bond_set:
            continue
        d = float(np.linalg.norm(coords[i] - coords[j]))
        if d < min_nonbonded:
            min_nonbonded = d
            min_pair = (i, j)
        if d < 0.9:
            contacts_under_09.append((i, j, d))

    counts = formula(atoms)
    extent = coords.max(axis=0) - coords.min(axis=0)
    report = {
        "atom_count": len(atoms),
        "formula": counts,
        "formula_string": formula_string(counts),
        "bond_count": len(bonds),
        "bond_length_min_A": float(bond_lengths.min()),
        "bond_length_max_A": float(bond_lengths.max()),
        "bond_length_mean_A": float(bond_lengths.mean()),
        "minimum_nonbonded_distance_A": None if not math.isfinite(min_nonbonded) else min_nonbonded,
        "minimum_nonbonded_pair": None if min_pair is None else {
            "i": min_pair[0] + 1,
            "j": min_pair[1] + 1,
            "atom_i": f"{atoms[min_pair[0]].name}/{atoms[min_pair[0]].residue}/{elements[min_pair[0]]}",
            "atom_j": f"{atoms[min_pair[1]].name}/{atoms[min_pair[1]].residue}/{elements[min_pair[1]]}",
        },
        "nonbonded_contacts_below_0.9_A": len(contacts_under_09),
        "extent_A": {"x": float(extent[0]), "y": float(extent[1]), "z": float(extent[2])},
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", "--units", type=int, required=True, help="number of glucose units")
    ap.add_argument("--chain", choices=sorted(TEMPLATES), default="origin")
    ap.add_argument("--endcaps", choices=["oh", "h"], default="oh")
    ap.add_argument("--prefix", type=Path, help="output prefix; default cellulose_Ibeta_<chain>_N<N>_<endcaps>")
    ap.add_argument("--no-center", action="store_true", help="retain crystallographic absolute placement")
    ap.add_argument("--no-pdb", action="store_true")
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()

    prefix = args.prefix or Path(f"cellulose_Ibeta_{args.chain}_N{args.units}_{args.endcaps}")
    atoms, bonds = build_chain(args.units, args.chain, args.endcaps, center=not args.no_center)
    write_xyz(prefix.with_suffix(".xyz"), atoms, args.units, args.chain, args.endcaps)
    if not args.no_pdb:
        write_pdb(prefix.with_suffix(".pdb"), atoms, bonds, args.units, args.chain, args.endcaps)
    if not args.no_validate:
        report = validate(atoms, bonds)
        report.update({"N": args.units, "chain": args.chain, "endcaps": args.endcaps})
        prefix.with_name(prefix.name + "_validation").with_suffix(".json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
