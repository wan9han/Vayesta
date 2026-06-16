"""MFCC-style cap subtraction for covalent fragmentation.

Cutting a covalent bond (e.g. a polyethylene C–C bond) leaves dangling
valences. MFCC (Molecular Fractionation with Conjugate Caps, He et al.)
saturates each fragment with small *caps* so every fragment becomes a
closed-shell molecule, then subtracts the cap energies::

    E_MFCC = Σ_f E(fragment_f, capped) − Σ_c E(conjugate_cap_c)

This is a *linear* combination (no inclusion–exclusion): the caps simply
remove the energy that was double-counted at each cut. MFCC is approximate
— its accuracy vs. the full-system energy is the *cap error*, which Stage 1
measures against SIESTA. It is the covalent counterpart of MBE; the two
compose (MFCC repairs the cuts, MBE handles fragment–fragment many-body
interaction).

For Stage 0 we only assert the linear bookkeeping is correct (given defined
energies, the output equals the defined sum). Physical accuracy is Stage 1.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Tuple

Subsys = Tuple[int, ...]


def mfcc_total(
    fragment_energies: Mapping,
    cap_energies: Mapping,
) -> Dict[str, object]:
    """Compute ``Σ E(capped fragment) − Σ E(conjugate cap)``.

    Parameters
    ----------
    fragment_energies:
        Mapping ``fragment_id -> energy`` of the *capped* fragments.
    cap_energies:
        Mapping ``cap_id -> energy`` of the conjugate caps.

    Returns
    -------
        Dict with ``total``, ``fragment_sum``, ``cap_sum``,
        ``num_fragments`` and ``num_caps``. The combination is linear, so
        for a single cut it reduces to ``E_frag_a + E_frag_b − E_cap``.
    """
    frag_sum = float(sum(fragment_energies.values()))
    cap_sum = float(sum(cap_energies.values()))
    return {
        "total": frag_sum - cap_sum,
        "fragment_sum": frag_sum,
        "cap_sum": cap_sum,
        "num_fragments": len(fragment_energies),
        "num_caps": len(cap_energies),
        "formula": "sum(capped_fragment) - sum(conjugate_cap)",
    }


def capped_fragment_scheme(num_fragments: int) -> Tuple[Iterable[int], Iterable[Tuple[int, int]]]:
    """Return ``(fragment_ids, cut_pairs)`` for a linear chain split into
    ``num_fragments`` contiguous capped fragments.

    A chain cut into ``num_fragments`` pieces has ``num_fragments - 1`` cut
    sites, hence ``num_fragments - 1`` conjugate caps. This helper just
    states the topology; the actual cap chemistry/geometry is Stage 1.
    """
    fragment_ids = list(range(num_fragments))
    cut_pairs = [(i, i + 1) for i in range(num_fragments - 1)]
    return fragment_ids, cut_pairs
