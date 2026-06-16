"""Subsystem (monomer / dimer / trimer / ...) generation.

A *subsystem* is a subset of fragments, identified by a sorted tuple of
fragment indices, e.g. ``(0,)`` is a monomer, ``(0, 2)`` a dimer,
``(1, 2, 4)`` a trimer. Sorted-tuple keys are canonical and hashable.

Many-body expansion needs every subsystem's proper subsets to be present
too. The cutoff rule used here is the *clique* rule: a subsystem is kept
only if every pair of its fragments is within ``cutoff`` of each other. A
clique contains all its sub-cliques, so the subset requirement of the
inclusion–exclusion step is preserved.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Callable, Iterator, Optional, Sequence, Tuple

Subsys = Tuple[int, ...]


def _pair_within_cutoff(
    i: int,
    j: int,
    centers: Sequence[Sequence[float]],
    cutoff: float,
) -> bool:
    ci, cj = centers[i], centers[j]
    # Euclidean distance, arbitrary dimension (atoms are 3D; tests may use 1D).
    d2 = sum((a - b) ** 2 for a, b in zip(ci, cj))
    return math.sqrt(d2) <= cutoff


def _clique_ok(
    key: Subsys,
    centers: Sequence[Sequence[float]],
    cutoff: float,
) -> bool:
    for i, j in combinations(key, 2):
        if not _pair_within_cutoff(i, j, centers, cutoff):
            return False
    return True


def iter_subsystems(
    num_fragments: int,
    max_order: int,
    cutoff: Optional[float] = None,
    fragment_centers: Optional[Sequence[Sequence[float]]] = None,
) -> Iterator[Subsys]:
    """Yield subsystem keys for fragments ``0 .. num_fragments-1`` up to
    ``max_order`` fragments per subsystem.

    Parameters
    ----------
    num_fragments:
        Total number of fragments.
    max_order:
        Largest subsystem size to generate (1 = monomers only, 2 = up to
        dimers, 3 = up to trimers). Must satisfy ``1 <= max_order <= num_fragments``.
    cutoff, fragment_centers:
        Optional distance cutoff. When both are given, a subsystem is kept
        only if every fragment pair in it lies within ``cutoff`` (clique
        rule). ``fragment_centers[i]`` is fragment *i*'s centre coordinate.

    Yields
    ------
        Sorted-tuple subsystem keys, grouped by increasing size.
    """
    if num_fragments < 1:
        raise ValueError(f"num_fragments must be >= 1, got {num_fragments}")
    if not (1 <= max_order <= num_fragments):
        raise ValueError(
            f"max_order must satisfy 1 <= max_order <= num_fragments; "
            f"got max_order={max_order}, num_fragments={num_fragments}"
        )
    if (cutoff is None) != (fragment_centers is None):
        raise ValueError(
            "cutoff and fragment_centers must be given together (or both omitted)"
        )
    if fragment_centers is not None and len(fragment_centers) != num_fragments:
        raise ValueError(
            f"fragment_centers has length {len(fragment_centers)} but "
            f"num_fragments={num_fragments}"
        )

    for order in range(1, max_order + 1):
        for combo in combinations(range(num_fragments), order):
            if cutoff is not None and not _clique_ok(combo, fragment_centers, cutoff):
                continue
            yield combo


def full_system_key(num_fragments: int) -> Subsys:
    """The subsystem key representing all fragments together."""
    return tuple(range(num_fragments))
