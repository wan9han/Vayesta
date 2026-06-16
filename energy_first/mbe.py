"""Many-body expansion (inclusion–exclusion) of subsystem energies.

Given a backend that returns the energy of any subsystem (a subset of
fragments), the many-body expansion reconstructs the total energy as a sum
of *increments*::

    ΔE_i   = E_i
    ΔE_ij  = E_ij − E_i − E_j
    ΔE_ijk = E_ijk − E_ij − E_ik − E_jk + E_i + E_j + E_k
    ...

General recursive definition::

    ΔE_S = E_S − Σ_{∅ ≠ T ⊊ S} ΔE_T

and the truncated expansion to order ``k`` is::

    E_MBE(k) = Σ_{|S| ≤ k} ΔE_S

Key correctness property (verified by the unit tests): when
``max_order == num_fragments`` (no truncation, no cutoff),

    E_MBE(num_fragments) == E_full

to machine precision — *regardless* of what the backend returns. This is
what makes the formula unfakeable: a wrong implementation cannot pass it.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable, Dict, Optional, Tuple

from .subsystems import iter_subsystems

Subsys = Tuple[int, ...]
Backend = Callable[[Subsys], float]


def compute_increments(raw_energies: Dict[Subsys, float]) -> Dict[Subsys, float]:
    """Turn raw subsystem energies into many-body increments.

    Parameters
    ----------
    raw_energies:
        Mapping from subsystem key (sorted tuple) to its energy. For every
        key ``S`` present, **every** non-empty proper subset of ``S`` must
        also be a key — otherwise inclusion–exclusion is undefined and a
        ``KeyError`` is raised naming the missing subset.

    Returns
    -------
        Mapping ``S -> ΔE_S`` computed by the recursive definition above.
        Iteration is in increasing subsystem size so lower-order increments
        are always available when needed.
    """
    increments: Dict[Subsys, float] = {}
    for key in sorted(raw_energies, key=lambda k: (len(k), k)):
        n = len(key)
        lower_sum = 0.0
        for r in range(1, n):  # proper non-empty subsets, sizes 1..n-1
            for sub in combinations(key, r):
                if sub not in increments:
                    raise KeyError(
                        f"inclusion–exclusion needs subsystem {sub} "
                        f"(proper subset of {key}) but it is missing"
                    )
                lower_sum += increments[sub]
        increments[key] = float(raw_energies[key]) - lower_sum
    return increments


def total_energy(
    increments: Dict[Subsys, float],
    max_order: Optional[int] = None,
) -> float:
    """Sum increments with ``len(key) <= max_order`` (``None`` = all)."""
    total = 0.0
    for key, inc in increments.items():
        if max_order is None or len(key) <= max_order:
            total += inc
    return total


def many_body_expansion(
    backend: Backend,
    num_fragments: int,
    max_order: int,
    cutoff: Optional[float] = None,
    fragment_centers=None,
) -> Dict[str, object]:
    """Run a many-body expansion against ``backend``.

    ``backend(key)`` must return the energy of subsystem ``key`` (a sorted
    tuple of fragment indices). The function generates all subsystems up to
    ``max_order`` (subject to ``cutoff``), queries the backend, computes
    increments, and returns the truncated total.

    Returns a dict with ``subsystems``, ``raw_energies``, ``increments`` and
    ``total``.
    """
    subsystems = list(
        iter_subsystems(num_fragments, max_order, cutoff, fragment_centers)
    )
    raw_energies: Dict[Subsys, float] = {s: float(backend(s)) for s in subsystems}
    increments = compute_increments(raw_energies)
    return {
        "subsystems": subsystems,
        "raw_energies": raw_energies,
        "increments": increments,
        "total": total_energy(increments, max_order),
    }
