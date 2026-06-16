"""MBE(2) over capped fragments — drives the MFCC per-cut error down.

Plain MFCC (1-body) misses the interaction across each cut, giving ~1 eV/cut.
A 2-body many-body expansion over the capped segments restores it: for each
cut c between adjacent segments i, i+1, we evaluate the *joined* fragment
(both segments with the real bond at c restored, capped only at the outer
cuts) and form the increment

    Δ_c = E(M_{i,i+1}) − E(M_i) − E(M_{i+1}) + E(cap_c)

The +E(cap_c) un-subtracts the cap that no longer exists in the joined piece.
Summed over all cuts, the cap terms cancel exactly (every cut is shared by
exactly one adjacent pair), leaving

    E^(2) = Σ_i E(M_i) + Σ_c [ E(M_{i,i+1}) − E(M_i) − E(M_{i+1}) ]

which is inclusion–exclusion over overlapping joined pieces. The residual
error vs E(full) is the (small) 3-body term. This is the teacher-suggested
MBE refinement, built on the already-falsifiable 1-body MFCC.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np

from .mfcc_geometry import build_mfcc
from .molecule import Molecule


def _find_fragment_with(
    frags: List[Molecule], targets
) -> Molecule:
    """Return the fragment whose carbons include all ``targets`` (by coord match)."""
    targets = [np.asarray(t, dtype=float) for t in targets]
    for f in frags:
        carb_coords = np.array(
            [c for el, c in zip(f.elements, f.coords) if el == "C"], dtype=float
        )
        if len(carb_coords) == 0:
            continue
        ok = True
        for t in targets:
            if np.min(np.linalg.norm(carb_coords - t, axis=1)) > 1e-4:
                ok = False
                break
        if ok:
            return f
    raise ValueError("no fragment contains all target carbons")


def mbe2_over_capped_fragments(
    mol: Molecule,
    cuts: List[Tuple[int, int]],
    run_func: Callable[[Molecule], float],
) -> Dict[str, object]:
    """Compute 1-body (MFCC) and 2-body (MBE) energies over capped segments.

    ``run_func(molecule) -> energy`` should be cached (identical fragments are
    queried repeatedly). Returns ``E_mbe1``, ``E_mbe2``, per-cut increments and
    the number of distinct SIESTA jobs.
    """
    cuts = [tuple(c) for c in cuts]

    # 1-body
    frags1, caps1 = build_mfcc(mol, cuts)
    e_frags1 = [run_func(f) for f in frags1]
    e_caps1 = [run_func(c) for c in caps1]
    e_mbe1 = sum(e_frags1) - sum(e_caps1)

    # 2-body: join the two segments sharing each cut
    increments = []
    for k, cut in enumerate(cuts):
        ci, cj = mol.coords[cut[0]], mol.coords[cut[1]]
        reduced = cuts[:k] + cuts[k + 1:]
        frags2, _ = build_mfcc(mol, reduced)
        merged = _find_fragment_with(frags2, [ci, cj])
        e_merged = run_func(merged)

        m_i = _find_fragment_with(frags1, [ci])
        m_j = _find_fragment_with(frags1, [cj])
        e_mi = run_func(m_i)
        e_mj = run_func(m_j)
        e_cap_k = e_caps1[k]

        delta = e_merged - e_mi - e_mj + e_cap_k
        increments.append({"cut": list(cut), "delta_ev": delta})

    e_mbe2 = e_mbe1 + sum(d["delta_ev"] for d in increments)
    return {
        "num_cuts": len(cuts),
        "E_mbe1_ev": e_mbe1,
        "E_mbe2_ev": e_mbe2,
        "increments_ev": increments,
    }
