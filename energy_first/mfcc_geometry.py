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


def _carbon_graph(mol: Molecule, removed_edges):
    """Adjacency among carbons with ``removed_edges`` (a set of frozensets) deleted."""
    adj = {c: set() for c in carbon_indices(mol)}
    for i, j in bonded_carbons(mol):
        if frozenset((i, j)) in removed_edges:
            continue
        adj[i].add(j)
        adj[j].add(i)
    return adj


def _segments(mol: Molecule, cuts) -> List[List[int]]:
    """Connected carbon components after removing all cut edges (chain → segments)."""
    removed = {frozenset(c) for c in cuts}
    adj = _carbon_graph(mol, removed)
    seen: set = set()
    segs: List[List[int]] = []
    for start in carbon_indices(mol):
        if start in seen:
            continue
        comp: List[int] = []
        dq = deque([start])
        seen.add(start)
        while dq:
            n = dq.popleft()
            comp.append(n)
            for m in adj[n]:
                if m not in seen:
                    seen.add(m)
                    dq.append(m)
        segs.append(sorted(comp))
    return segs


def _cap_positions(mol: Molecule, cuts, ch_bond: float):
    """Map each cut-endpoint carbon to its cap-H coordinate (pointing at the partner)."""
    cap_pos = {}
    for i, j in cuts:
        pi, pj = mol.coords[i], mol.coords[j]
        d = _dist(pi, pj)
        if d == 0:
            raise ValueError(f"cut carbons {(i, j)} coincide")
        u = (pj - pi) / d
        cap_pos[i] = pi + u * ch_bond
        cap_pos[j] = pj - u * ch_bond
    return cap_pos


def build_mfcc(
    mol: Molecule,
    cuts,
    ch_bond: float = CH_BOND,
) -> Tuple[List[Molecule], List[Molecule]]:
    """General MFCC build for one or more C–C cuts.

    ``cuts`` is a list of ``(i, j)`` bonded-carbon pairs. The chain is split
    into ``len(cuts)+1`` contiguous capped fragments, and one H₂ conjugate
    cap is produced per cut. Each cap hydrogen occupies the same coordinate
    in its fragment and in the conjugate cap, so the cap bookkeeping cancels.
    """
    cuts = [tuple(c) for c in cuts]
    segs = _segments(mol, cuts)
    cap_pos = _cap_positions(mol, cuts, ch_bond)

    fragments: List[Molecule] = []
    for seg in segs:
        atoms = list(seg)
        for c in seg:
            atoms.extend(hydrogens_bonded_to(mol, c))
        els = [mol.elements[a] for a in atoms]
        coords = mol.coords[atoms].copy()
        frag = Molecule(els, coords, label=f"frag_{seg[0]}")
        for c in seg:  # append cap H for every cut endpoint in this segment
            if c in cap_pos:
                frag.append("H", cap_pos[c])
        fragments.append(frag)

    caps = [
        Molecule(["H", "H"], np.vstack([cap_pos[i], cap_pos[j]]), label=f"cap_{i}_{j}")
        for i, j in cuts
    ]
    return fragments, caps


def build_mfcc_cut(
    mol: Molecule,
    cut: Tuple[int, int],
    ch_bond: float = CH_BOND,
) -> Tuple[List[Molecule], List[Molecule]]:
    """Single-cut MFCC (convenience wrapper over :func:`build_mfcc`)."""
    return build_mfcc(mol, [cut], ch_bond=ch_bond)


def _chain_bonds(mol: Molecule):
    """Bonds between adjacent carbons, keyed by lower carbon-list rank."""
    cs = carbon_indices(mol)
    rank = {c: k for k, c in enumerate(cs)}
    chain = {}
    for i, j in bonded_carbons(mol):
        if abs(rank[i] - rank[j]) == 1:
            k = min(rank[i], rank[j])
            chain[k] = (i, j)
    return chain


def pick_cuts(mol: Molecule, num_fragments: int) -> List[Tuple[int, int]]:
    """Choose cuts that split the carbon chain into ``num_fragments`` near-equal segments."""
    cs = carbon_indices(mol)
    n = len(cs)
    if num_fragments < 1 or num_fragments > n:
        raise ValueError(f"num_fragments={num_fragments} out of range [1,{n}]")
    if num_fragments == 1:
        return []
    chain = _chain_bonds(mol)
    cuts = []
    for f in range(1, num_fragments):
        k = round(f * n / num_fragments)  # boundary rank; cut bond cs[k-1]--cs[k]
        cuts.append(chain[k - 1])
    return cuts


def pick_middle_cut(mol: Molecule) -> Tuple[int, int]:
    """Choose the central C–C bond as the single cut (symmetric choice)."""
    cuts = pick_cuts(mol, 2)
    return cuts[0]
