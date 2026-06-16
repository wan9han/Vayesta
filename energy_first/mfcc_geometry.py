"""MFCC cut + cap geometry for a covalent (polyethylene) chain.

Cutting a C–C bond leaves two dangling valences. We repair each side with a
hydrogen cap placed along the cut bond, turning each fragment into a
closed-shell molecule, and we keep the two cap hydrogens as a conjugate-cap
H₂ molecule evaluated at the *same* cap positions (so the cap contribution
cancels consistently between fragment and conjugate cap).

This is the geometry half of MFCC; the energy combination lives in
:mod:`energy_first.mfcc`.
"""

from __future__ import annotations

from collections import deque
from typing import List, Sequence, Tuple

import numpy as np

from .molecule import Molecule

# Connectivity thresholds (Å). C–C single bond ≈ 1.53, C–H ≈ 1.10.
CC_MAX = 1.70
CH_MAX = 1.30
CH_BOND = 1.10


def carbon_indices(mol: Molecule) -> List[int]:
    return [i for i, el in enumerate(mol.elements) if el == "C"]


def _dist(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def bonded_carbons(mol: Molecule, cc_max: float = CC_MAX) -> List[Tuple[int, int]]:
    """Pairs of carbon atom indices closer than ``cc_max``."""
    cs = carbon_indices(mol)
    pairs: List[Tuple[int, int]] = []
    for a in range(len(cs)):
        for b in range(a + 1, len(cs)):
            if _dist(mol.coords[cs[a]], mol.coords[cs[b]]) <= cc_max:
                pairs.append((cs[a], cs[b]))
    return pairs


def hydrogens_bonded_to(mol: Molecule, carbon_idx: int, ch_max: float = CH_MAX) -> List[int]:
    """H atom indices within ``ch_max`` of ``carbon_idx``."""
    out = []
    for i, el in enumerate(mol.elements):
        if el == "H" and _dist(mol.coords[i], mol.coords[carbon_idx]) <= ch_max:
            out.append(i)
    return out


def _sides_of_cut(mol: Molecule, cut: Tuple[int, int]) -> Tuple[List[int], List[int]]:
    """Split carbons into two sides by removing the cut edge from the C–C graph."""
    adj = {c: set() for c in carbon_indices(mol)}
    for i, j in bonded_carbons(mol):
        if {i, j} == set(cut):
            continue  # remove the cut edge
        adj[i].add(j)
        adj[j].add(i)

    def bfs(start) -> List[int]:
        seen = {start}
        dq = deque([start])
        while dq:
            n = dq.popleft()
            for m in adj[n]:
                if m not in seen:
                    seen.add(m)
                    dq.append(m)
        return sorted(seen)

    return bfs(cut[0]), bfs(cut[1])


def build_mfcc_cut(
    mol: Molecule,
    cut: Tuple[int, int],
    ch_bond: float = CH_BOND,
) -> Tuple[List[Molecule], List[Molecule]]:
    """Build capped fragments and conjugate cap(s) for a single C–C cut.

    Parameters
    ----------
    mol:
        The full molecule.
    cut:
        ``(i, j)`` mol-atom indices of the two bonded carbons to cut.
    ch_bond:
        C–H bond length (Å) used to place the cap hydrogen.

    Returns
    -------
    (fragments, caps)
        ``fragments`` has one capped :class:`Molecule` per side; ``caps`` has
        one H₂ conjugate cap. Each cap hydrogen sits at the same position in
        its fragment and in the conjugate cap, so the cap bookkeeping cancels.
    """
    i, j = cut
    side_a, side_b = _sides_of_cut(mol, cut)

    pi, pj = mol.coords[i], mol.coords[j]
    dij = _dist(pi, pj)
    if dij == 0:
        raise ValueError(f"cut carbons {cut} coincide")
    u_ij = (pj - pi) / dij  # unit vector from i to j

    cap_h_a = pi + u_ij * ch_bond      # caps carbon i, pointing toward j
    cap_h_b = pj - u_ij * ch_bond      # caps carbon j, pointing toward i

    def build_side(side_carbons, cap_carbon, cap_h) -> Molecule:
        atoms = list(side_carbons)
        for c in side_carbons:
            atoms.extend(hydrogens_bonded_to(mol, c))
        els = [mol.elements[a] for a in atoms]
        coords = mol.coords[atoms].copy()
        frag = Molecule(els, coords, label=f"frag_{cap_carbon}")
        frag.append("H", cap_h)
        return frag

    frag_a = build_side(side_a, i, cap_h_a)
    frag_b = build_side(side_b, j, cap_h_b)
    cap = Molecule(["H", "H"], np.vstack([cap_h_a, cap_h_b]), label="cap")

    return [frag_a, frag_b], [cap]


def pick_middle_cut(mol: Molecule) -> Tuple[int, int]:
    """Choose the central C–C bond of the carbon chain as the cut.

    For ``n`` carbons this cuts between carbon-list indices ``n//2 - 1`` and
    ``n//2``. For a single cut this is the natural symmetric choice.
    """
    cs = carbon_indices(mol)
    if len(cs) < 2:
        raise ValueError("need at least 2 carbons to cut")
    bonds = bonded_carbons(mol)
    # order bonds by their position along the chain (by carbon-list index sum)
    rank = {c: k for k, c in enumerate(cs)}
    bonds_sorted = sorted(bonds, key=lambda b: rank[b[0]] + rank[b[1]])
    return bonds_sorted[len(bonds_sorted) // 2]
