"""Controllable mock backend for formula verification.

A mock backend returns energies we *define*, so the combination formulas
can be checked against known answers. Three behaviours are supported:

* ``additive``     : ``E(S) = Σ_{i∈S} e_i``                 → MBE(1) recovers total
* ``pairwise``     : ``E(S) = Σ e_i + Σ_{i<j∈S} v_ij``       → MBE(2) recovers total
* arbitrary table  : ``E(S) = table[S]``                      → MBE(N) recovers total

The arbitrary-table mode is the exactness workhorse: feed it any numbers
and MBE at full order must reproduce ``table[(0,1,...,N-1)]`` to ~1e-12.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional, Tuple

Subsys = Tuple[int, ...]


class MockBackend:
    """Callable backend returning defined subsystem energies."""

    def __init__(
        self,
        table: Optional[Dict[Subsys, float]] = None,
        *,
        additive: Optional[Iterable[float]] = None,
        pairwise: Optional[Dict[Tuple[int, int], float]] = None,
        callable_: Optional[Callable[[Subsys], float]] = None,
    ):
        provided = [
            table is not None,
            additive is not None,
            callable_ is not None,
        ]
        if sum(bool(x) for x in provided) != 1:
            raise ValueError(
                "provide exactly one of: table, additive, or callable_"
            )
        self._table = table
        self._additive = list(additive) if additive is not None else None
        self._pairwise = dict(pairwise) if pairwise is not None else {}
        self._callable = callable_
        self.calls: list[Subsys] = []

    def __call__(self, key: Subsys) -> float:
        self.calls.append(key)
        if self._table is not None:
            if key not in self._table:
                raise KeyError(f"mock table has no entry for subsystem {key}")
            return float(self._table[key])
        if self._additive is not None:
            e = 0.0
            for i in key:
                e += self._additive[i]
            for i, j in self._pairs(key):
                e += self._pairwise.get((i, j), 0.0)
            return e
        # callable_
        return float(self._callable(key))

    @staticmethod
    def _pairs(key: Subsys):
        n = len(key)
        for a in range(n):
            for b in range(a + 1, n):
                yield tuple(sorted((key[a], key[b])))
