"""Pure-geometry tests for the MFCC cut + cap builder (no SIESTA)."""

import math

import numpy as np

from energy_first.mfcc_geometry import (
    bonded_carbons,
    build_mfcc_cut,
    carbon_indices,
    hydrogens_bonded_to,
    pick_middle_cut,
)
from energy_first.molecule import Molecule


def _ethane_like() -> Molecule:
    # C0 at origin, C1 at 1.53 on x; 3 H's on each carbon.
    els = ["C", "C", "H", "H", "H", "H", "H", "H"]
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.53, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -0.5, 0.866],
            [0.0, -0.5, -0.866],
            [1.53, 1.0, 0.0],
            [1.53, -0.5, 0.866],
            [1.53, -0.5, -0.866],
        ]
    )
    return Molecule(els, coords, "ethane_test")


def test_carbon_and_hydrogen_connectivity():
    mol = _ethane_like()
    assert carbon_indices(mol) == [0, 1]
    assert bonded_carbons(mol) == [(0, 1)]
    assert sorted(hydrogens_bonded_to(mol, 0)) == [2, 3, 4]
    assert sorted(hydrogens_bonded_to(mol, 1)) == [5, 6, 7]


def test_ethane_cut_gives_two_methanes_plus_h2():
    mol = _ethane_like()
    frags, caps = build_mfcc_cut(mol, cut=(0, 1))
    assert len(frags) == 2 and len(caps) == 1
    for f in frags:
        # methane: 1 C + 4 H
        assert f.elements.count("C") == 1
        assert f.elements.count("H") == 4
        assert f.natoms == 5
    cap = caps[0]
    assert cap.elements == ["H", "H"]


def test_cap_hydrogen_positions_along_cut_bond():
    mol = _ethane_like()
    frags, caps = build_mfcc_cut(mol, cut=(0, 1), ch_bond=1.10)
    cap = caps[0]
    # cap H's at x = 1.10 (on C0 side) and x = 1.53-1.10 = 0.43 (on C1 side)
    xs = sorted(round(c[0], 6) for c in cap.coords)
    assert math.isclose(xs[0], 0.43, abs_tol=1e-6)
    assert math.isclose(xs[1], 1.10, abs_tol=1e-6)
    # the same cap-H positions appear in the fragments (consistency → cancellation)
    frag_cap_xs = sorted(round(f.coords[-1, 0], 6) for f in frags)
    assert frag_cap_xs == xs


def test_butane_cut_gives_two_ethanes():
    # C0-C1-C2-C3 linear carbons; minimal H's to satisfy connectivity only.
    els = ["C", "C", "C", "C"]
    coords = np.array(
        [[0, 0, 0], [1.53, 0, 0], [3.06, 0, 0], [4.59, 0, 0]],
        dtype=float,
    )
    mol = Molecule(els, coords, "butane_test")
    cut = pick_middle_cut(mol)  # expect the central bond (1,2)
    assert set(cut) == {1, 2}
    frags, caps = build_mfcc_cut(mol, cut)
    for f in frags:
        # each side has 2 carbons; + 1 cap H → C2H (no real H's in this stub)
        assert f.elements.count("C") == 2
        assert f.elements.count("H") == 1
    assert len(caps) == 1 and caps[0].elements == ["H", "H"]


def test_sides_partition_all_carbons():
    mol = _ethane_like()
    frags, _ = build_mfcc_cut(mol, cut=(0, 1))
    # the two fragments' carbons together cover both original carbons, disjoint
    cs = []
    for f in frags:
        cs += [i for i, e in enumerate(f.elements) if e == "C"]
    assert len(cs) == 2  # one carbon per fragment
