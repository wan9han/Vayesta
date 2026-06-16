"""Unit tests for the MFCC cap-subtraction combination (linear bookkeeping)."""

import math

from energy_first.mfcc import capped_fragment_scheme, mfcc_total


def test_mfcc_single_cut_is_frag_sum_minus_cap():
    # Ethane-like: one C–C cut → two capped fragments + one conjugate cap.
    out = mfcc_total(
        fragment_energies={0: -40.0, 1: -40.0},  # two capped CH3 fragments
        cap_energies={(0, 1): -1.0},             # one conjugate cap
    )
    assert math.isclose(out["total"], -79.0, abs_tol=1e-12)
    assert math.isclose(out["fragment_sum"], -80.0, abs_tol=1e-12)
    assert math.isclose(out["cap_sum"], -1.0, abs_tol=1e-12)
    assert out["num_fragments"] == 2
    assert out["num_caps"] == 1


def test_mfcc_linear_chain_topology():
    # A chain of N contiguous capped fragments has N-1 cut sites → N-1 caps.
    fragment_ids, cut_pairs = capped_fragment_scheme(num_fragments=4)
    assert fragment_ids == [0, 1, 2, 3]
    assert cut_pairs == [(0, 1), (1, 2), (2, 3)]
    assert len(cut_pairs) == 3


def test_mfcc_two_cuts():
    out = mfcc_total(
        fragment_energies={0: -10.0, 1: -12.0, 2: -10.0},
        cap_energies={(0, 1): -0.5, (1, 2): -0.5},
    )
    # (-10 -12 -10) - (-0.5 -0.5) = -32 + 1 = -31
    assert math.isclose(out["total"], -31.0, abs_tol=1e-12)


def test_mfcc_formula_field_documented():
    out = mfcc_total(fragment_energies={0: 1.0}, cap_energies={})
    assert out["formula"] == "sum(capped_fragment) - sum(conjugate_cap)"
