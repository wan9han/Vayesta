"""Vectorized polyethylene (PE) chain generator.

Drop-in, geometry-identical replacement for the per-atom Python loops in
``gen.py``: the zigzag carbon backbone and the two side hydrogens per carbon
are closed-form functions of the carbon index, so the whole chain is built
with a handful of numpy array ops instead of ~N Python ``append`` calls.

Returns ``(elements, coords)`` as numpy arrays directly -- no list-of-dicts,
no FDF text -- so callers (e.g. ``weak_scale_pe``) can use the chain in
process and skip the 12.7 GB serialize/parse round-trip at full scale.

Geometry mirrors ``gen.py`` exactly (C-C zigzag along x, 2 H per carbon at
+/- hz_base with y offset by sign*hy_base, 2 end-cap H via the virtual-carbon
formula), then centered to the bounding-box center.
"""

from __future__ import annotations

import numpy as np


def generate_pe_chain(n_carbons: int, cc_bond_length: float = 1.54,
                      ch_bond_length: float = 1.09):
    """Build a C_n H_{2n+2} polyethylene chain, vectorized.

    Defaults (cc=1.54, ch=1.09) match ``gen.py``'s CLI defaults (the values
    the pipeline actually uses via the subprocess), NOT generate_polyethylene's
    function defaults (1.53/1.10).

    Returns ``(elements, coords)``:
      - elements: np.ndarray of 'C'/'H' (n_carbons C, then 2*n_carbons+2 H)
      - coords:   (n_atoms, 3) float64 array, centered to bbox center
    """
    theta = 109.471 * np.pi / 180.0
    phi = theta / 2.0
    dx = cc_bond_length * np.sin(phi)
    dy = cc_bond_length * np.cos(phi)
    hy_base = 0.632 * (ch_bond_length / 1.07)
    hz_base = 0.863 * (ch_bond_length / 1.07)

    # --- 1. carbon backbone (zigzag along x) ---
    i = np.arange(n_carbons)
    cx = i * dx
    cy = np.where(i % 2 == 0, -dy / 2.0, dy / 2.0)
    cz = np.zeros(n_carbons)

    # --- 2. side hydrogens (2 per carbon: z+ then z-) ---
    sign = np.where(cy > 0, 1.0, -1.0)
    hy = cy + sign * hy_base
    hcx = np.repeat(cx, 2)
    hcy = np.repeat(hy, 2)
    hcz = np.empty(2 * n_carbons)
    hcz[0::2] = hz_base
    hcz[1::2] = -hz_base

    # --- 3. end-cap hydrogens (2, scalar; virtual-carbon formula from gen.py) ---
    c0x, c0y = float(cx[0]), float(cy[0])
    vx, vy = (c0x - dx) - c0x, (-c0y) - c0y
    nrm = np.sqrt(vx * vx + vy * vy)
    hsx = c0x + (vx / nrm) * ch_bond_length
    hsy = c0y + (vy / nrm) * ch_bond_length
    cnx, cny = float(cx[-1]), float(cy[-1])
    vx2, vy2 = (cnx + dx) - cnx, (-cny) - cny
    nrm2 = np.sqrt(vx2 * vx2 + vy2 * vy2)
    hex_ = cnx + (vx2 / nrm2) * ch_bond_length
    hey = cny + (vy2 / nrm2) * ch_bond_length

    # --- assemble (carbons, then side H, then 2 end caps; order is irrelevant
    #     downstream -- cKDTree/h_by_carbon/block-build are coord-set based) ---
    coords = np.vstack([
        np.column_stack([cx, cy, cz]),
        np.column_stack([hcx, hcy, hcz]),
        np.array([[hsx, hsy, 0.0], [hex_, hey, 0.0]]),
    ])
    elements = np.array(["C"] * n_carbons + ["H"] * (2 * n_carbons + 2))

    # --- 4. center to bounding-box center (matches gen.py) ---
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    coords = coords - (lo + hi) / 2.0
    return elements, coords
