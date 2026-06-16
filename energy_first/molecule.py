"""Molecule representation and SIESTA FDF I/O for the energy-first prototype.

A :class:`Molecule` is just elements + Cartesian coordinates (Å). We parse
the polyethylene FDF emitted by ``testcases/gen.py`` and we write a SIESTA
FDF with an *explicit* orthorhombic vacuum cell.

The explicit cell matters for a fair MFCC comparison: the full molecule and
every fragment/cap must be computed under the SAME numerical conditions.
Letting SIESTA auto-size the cell per molecule would inject a non-fragmentation
energy bias. We therefore wrap each molecule in ``bbox + cell_margin`` Å of
vacuum (orthorhombic), which makes periodic-image effects negligible and
uniform across runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Element -> atomic number (extend if polyethylene grows beyond C/H).
_ATOMIC_NUMBER: Dict[str, int] = {"H": 1, "C": 6}


@dataclass
class Molecule:
    elements: List[str]
    coords: np.ndarray
    label: str = "mol"

    def __post_init__(self) -> None:
        self.coords = np.asarray(self.coords, dtype=float).reshape(-1, 3)
        if len(self.elements) != len(self.coords):
            raise ValueError(
                f"elements ({len(self.elements)}) and coords ({len(self.coords)}) "
                "length mismatch"
            )

    @property
    def natoms(self) -> int:
        return len(self.elements)

    def copy(self) -> "Molecule":
        return Molecule(list(self.elements), self.coords.copy(), self.label)

    def append(self, element: str, coord) -> None:
        self.elements.append(element)
        self.coords = np.vstack([self.coords, np.asarray(coord, dtype=float).reshape(1, 3)])

    def bbox(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.coords.min(axis=0), self.coords.max(axis=0)


def parse_gen_fdf_text(text: str, label: str = "mol") -> Molecule:
    """Parse the FDF text produced by ``testcases/gen.py`` into a Molecule."""
    spec: Dict[int, str] = {}
    m = re.search(
        r"%block\s+ChemicalSpeciesLabel\s*\n(.*?)%endblock\s+ChemicalSpeciesLabel",
        text,
        re.S,
    )
    if not m:
        raise ValueError("ChemicalSpeciesLabel block not found")
    for line in m.group(1).splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0].isdigit():
            spec[int(parts[0])] = parts[2]

    rows: List[Tuple[float, float, float, int]] = []
    m = re.search(
        r"%block\s+AtomicCoordinatesAndAtomicSpecies\s*\n(.*?)%endblock\s+AtomicCoordinatesAndAtomicSpecies",
        text,
        re.S,
    )
    if not m:
        raise ValueError("AtomicCoordinatesAndAtomicSpecies block not found")
    for line in m.group(1).splitlines():
        parts = line.split()
        if len(parts) >= 4:
            rows.append((float(parts[0]), float(parts[1]), float(parts[2]), int(parts[3])))

    elements = [spec[r[3]] for r in rows]
    coords = np.array([[r[0], r[1], r[2]] for r in rows], dtype=float)
    return Molecule(elements, coords, label)


def _species_table(elements: List[str]) -> Tuple[List[Tuple[int, int, str]], Dict[str, int]]:
    """Build a ChemicalSpeciesLabel table; return (rows, element->species_index)."""
    uniq = sorted(set(elements), key=lambda e: _ATOMIC_NUMBER.get(e, 99))
    index: Dict[str, int] = {e: i + 1 for i, e in enumerate(uniq)}
    rows = [(index[e], _ATOMIC_NUMBER.get(e, 0), e) for e in uniq]
    return rows, index


def write_siesta_fdf(
    mol: Molecule,
    path,
    cell_margin: float = 8.0,
    mesh_cutoff_ry: float = 100.0,
    basis_size: str = "SZ",
    max_scf: int = 100,
) -> None:
    """Write a SIESTA FDF for ``mol`` with an explicit vacuum cell.

    Settings mirror ``gen.py`` (SZ / LDA-PZ / MeshCutoff) but add an explicit
    orthorhombic cell and use plain ``Diagonali`` (fine for the small
    molecules of Stage 1). Identical settings for full + fragments + caps keep
    the comparison bias-free.
    """
    from pathlib import Path

    path = Path(path)
    rows, index = _species_table(mol.elements)
    lo, hi = mol.bbox()
    cell = (hi - lo) + 2.0 * cell_margin  # orthorhombic vacuum box

    lines: List[str] = []
    lines.append(f"SystemLabel      {mol.label}")
    lines.append(f"NumberOfAtoms    {mol.natoms}")
    lines.append(f"NumberOfSpecies  {len(rows)}")
    lines.append("%block ChemicalSpeciesLabel")
    for sp, z, el in rows:
        lines.append(f"    {sp}    {z}  {el}")
    lines.append("%endblock ChemicalSpeciesLabel")
    lines.append("")
    lines.append("AtomicCoordinatesFormat NotScaledCartesianAng")
    lines.append("%block AtomicCoordinatesAndAtomicSpecies")
    for el, (x, y, z) in zip(mol.elements, mol.coords):
        lines.append(f"    {x:18.12f}   {y:18.12f}   {z:18.12f}    {index[el]}")
    lines.append("%endblock AtomicCoordinatesAndAtomicSpecies")
    lines.append("")
    lines.append("LatticeConstant 1.0 Ang")
    lines.append("%block LatticeVectors")
    lines.append(f"   {cell[0]:18.12f}   0   0")
    lines.append(f"   0   {cell[1]:18.12f}   0")
    lines.append(f"   0   0   {cell[2]:18.12f}")
    lines.append("%endblock LatticeVectors")
    lines.append("")
    lines.append("PAO.BasisType    split")
    lines.append(f"PAO.BasisSize    {basis_size}")
    lines.append("SolutionMethod   Diagonali")
    lines.append("PAO.SplitNorm    0.150000")
    lines.append("PAO.EnergyShift  0.020000  Ry")
    lines.append("Harris_functional false")
    lines.append("XC.functional    LDA")
    lines.append("XC.Authors       PZ")
    lines.append("SpinPolarized    false")
    lines.append(f"MeshCutoff       {mesh_cutoff_ry:.6f} Ry")
    lines.append("kgrid_cutoff     0.0 Bohr")
    lines.append("ElectronicTemperature 300.0 K")
    lines.append(f"MaxSCFIterations {max_scf}")
    lines.append("DM.NumberPulay   6")
    lines.append("DM.MixingWeight  0.050000")
    lines.append("UseSaveData      false")
    lines.append("WriteDM          false")
    lines.append("SaveHS           false")
    lines.append("WriteCoorXmol    true")
    path.write_text("\n".join(lines) + "\n")
