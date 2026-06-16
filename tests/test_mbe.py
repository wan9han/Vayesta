"""Unit tests for the many-body expansion formula.

The crown-jewel test is ``test_full_order_exactly_recovers_total``: with an
arbitrary mock energy table, MBE at full order must reproduce the table's
all-fragment energy to ~1e-12. This cannot be faked — a buggy formula fails
it for generic inputs.
"""

import math

from energy_first import MockBackend, many_body_expansion, total_energy
from energy_first.mbe import compute_increments


def test_monomer_only_equals_sum_of_monomers():
    table = {(0,): -76.0, (1,): -75.5, (2,): -76.2}
    res = many_body_expansion(MockBackend(table), num_fragments=3, max_order=1)
    assert math.isclose(res["total"], -76.0 - 75.5 - 76.2, abs_tol=1e-12)


def test_pairwise_mock_mb2_recovers_full_and_trimer_increment_zero():
    # Pure two-body interaction: trimers carry no genuine 3-body term,
    # so the trimer increment must vanish and MBE(2) must equal MBE(3).
    additive = [-76.0, -75.5, -76.2]
    pairwise = {(0, 1): -0.01, (0, 2): -0.02, (1, 2): -0.03}
    be = MockBackend(additive=additive, pairwise=pairwise)

    res2 = many_body_expansion(be, num_fragments=3, max_order=2)
    res3 = many_body_expansion(be, num_fragments=3, max_order=3)

    full = additive[0] + additive[1] + additive[2] + sum(pairwise.values())
    assert math.isclose(res2["total"], full, abs_tol=1e-12)
    assert math.isclose(res3["total"], full, abs_tol=1e-12)
    # The trimer increment is exactly zero for a pairwise-only model.
    assert math.isclose(res3["increments"][(0, 1, 2)], 0.0, abs_tol=1e-12)


def test_dimer_increment_matches_closed_form():
    table = {(0,): -76.0, (1,): -75.5, (0, 1): -152.01}
    res = many_body_expansion(MockBackend(table), num_fragments=2, max_order=2)
    # ΔE_01 = E_01 - E_0 - E_1
    expected = -152.01 - (-76.0) - (-75.5)
    assert math.isclose(res["increments"][(0, 1)], expected, abs_tol=1e-12)


def test_trimer_increment_matches_closed_form():
    table = {
        (0,): -76.0, (1,): -75.5, (2,): -76.2,
        (0, 1): -152.01, (0, 2): -152.02, (1, 2): -152.03,
        (0, 1, 2): -228.08,
    }
    res = many_body_expansion(MockBackend(table), num_fragments=3, max_order=3)
    # ΔE_012 = E_012 - E_01 - E_02 - E_12 + E_0 + E_1 + E_2
    expected = (
        table[(0, 1, 2)]
        - table[(0, 1)] - table[(0, 2)] - table[(1, 2)]
        + table[(0,)] + table[(1,)] + table[(2,)]
    )
    assert math.isclose(res["increments"][(0, 1, 2)], expected, abs_tol=1e-12)


def test_full_order_exactly_recovers_total():
    # Arbitrary numbers, including a genuine 3-body term. MBE at full order
    # MUST reproduce the table's all-fragment energy to machine precision.
    table = {
        (0,): -10.3, (1,): -20.7, (2,): -30.1, (3,): -40.9,
        (0, 1): -31.5, (0, 2): -40.8, (0, 3): -51.7,
        (1, 2): -51.2, (1, 3): -62.0, (2, 3): -71.6,
        (0, 1, 2): -62.4, (0, 1, 3): -73.1, (0, 2, 3): -83.9, (1, 2, 3): -93.5,
        (0, 1, 2, 3): -104.77,
    }
    res = many_body_expansion(MockBackend(table), num_fragments=4, max_order=4)
    assert math.isclose(res["total"], table[(0, 1, 2, 3)], abs_tol=1e-12)
    # Lower orders must NOT (in general) equal the full energy — truncation
    # leaves a non-zero remainder once a genuine 4-body term is present.
    res3 = many_body_expansion(MockBackend(table), num_fragments=4, max_order=3)
    assert not math.isclose(res3["total"], table[(0, 1, 2, 3)], abs_tol=1e-9)


def test_total_energy_helper_respects_max_order():
    table = {(0,): 1.0, (1,): 2.0, (0, 1): 4.0}
    inc = compute_increments(table)
    # ΔE_0=1, ΔE_1=2, ΔE_01 = 4-1-2 = 1
    assert math.isclose(total_energy(inc, max_order=1), 3.0, abs_tol=1e-12)
    assert math.isclose(total_energy(inc, max_order=2), 4.0, abs_tol=1e-12)
    assert math.isclose(total_energy(inc), 4.0, abs_tol=1e-12)


def test_additive_model_has_zero_higher_increments():
    additive = [-1.0, -2.0, -3.0]
    be = MockBackend(additive=additive)
    res = many_body_expansion(be, num_fragments=3, max_order=3)
    # No interactions → all dimer/trimer increments are zero.
    for key, inc in res["increments"].items():
        if len(key) >= 2:
            assert math.isclose(inc, 0.0, abs_tol=1e-12), (key, inc)
    assert math.isclose(res["total"], -6.0, abs_tol=1e-12)


def test_missing_subset_raises():
    # (0,1) present but its subset (1,) missing → clear error, not silence.
    table = {(0,): -1.0, (0, 1): -3.0}
    import pytest

    with pytest.raises(KeyError):
        compute_increments(table)
