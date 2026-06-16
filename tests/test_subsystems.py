"""Unit tests for subsystem generation and the distance cutoff (clique rule)."""

import pytest

from energy_first.subsystems import full_system_key, iter_subsystems


def test_keys_are_sorted_tuples_and_unique():
    keys = list(iter_subsystems(num_fragments=3, max_order=3))
    assert keys == [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
    assert len(set(keys)) == len(keys)
    for k in keys:
        assert list(k) == sorted(k)


def test_max_order_bounds():
    with pytest.raises(ValueError):
        list(iter_subsystems(num_fragments=3, max_order=0))
    with pytest.raises(ValueError):
        list(iter_subsystems(num_fragments=3, max_order=4))


def test_cutoff_and_centers_must_come_together():
    with pytest.raises(ValueError):
        list(iter_subsystems(num_fragments=3, max_order=2, cutoff=1.0))


def test_full_system_key():
    assert full_system_key(4) == (0, 1, 2, 3)


def test_cutoff_clique_rule_drops_far_subsystems():
    # Three fragments on a line at x = 0, 1, 5. Cutoff 2.0:
    # pairs (0,1) within, (0,2) and (1,2) beyond. Trimer dropped (not a clique).
    centers = [[0.0], [1.0], [5.0]]
    keys = list(iter_subsystems(3, max_order=3, cutoff=2.0, fragment_centers=centers))
    assert (0, 1) in keys
    assert (0, 2) not in keys
    assert (1, 2) not in keys
    assert (0, 1, 2) not in keys
    # monomers always present
    for i in range(3):
        assert (i,) in keys


def test_cutoff_keeps_subcliques():
    # Equilateral-ish: all pairs within cutoff → trimer kept, and so are all subsets.
    centers = [[0.0, 0.0], [1.0, 0.0], [0.5, 0.8]]
    keys = set(iter_subsystems(3, max_order=3, cutoff=2.0, fragment_centers=centers))
    assert (0, 1, 2) in keys
    assert (0, 1) in keys and (0, 2) in keys and (1, 2) in keys
