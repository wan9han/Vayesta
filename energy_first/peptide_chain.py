"""Polyglycine chain generator (simplest peptide) for MFCC generality validation.

Builds a polyglycine (-(NH-CH2-CO)-) backbone as a **planar zigzag** (like PE),
keeping y/z extent < 10 Å so the SIESTA mesh grid stays manageable. Side atoms
(carbonyl O, amide H, alpha H's, terminal caps) are placed from vector geometry.

The planar zigzag uses alternating turn directions at each backbone vertex so
the chain oscillates around the x-axis with small amplitude — same principle
as PE's zigzag, just with three different bond lengths/angles per residue.
"""

from __future__ import annotations

import numpy as np

_DEG = np.pi / 180.0


def generate_polyglycine(n_residues: int):
    """Build polyglycine C/H/N/O chain (planar zigzag backbone).

    Returns (elements, coords) numpy arrays centered to bbox center.
    """
    # Bond lengths
    r_NCa, r_CaC, r_CN = 1.46, 1.52, 1.33
    r_CO, r_NH, r_CaH, r_OH = 1.23, 1.01, 1.09, 0.96
    # Bond angles at each backbone vertex type (degrees)
    # Vertex pattern: N(123°), Cα(111°), C'(116°) — repeating
    bb_angles = [123.0, 111.0, 116.0]  # at N, Cα, C'
    bb_bonds  = [r_NCa, r_CaC, r_CN]   # N→Cα, Cα→C', C'→N(next)

    # ---- backbone: planar zigzag (all turns alternate up/down) ----
    # Place each backbone atom by turning (180-angle) from the incoming
    # direction, alternating side each step. Then remove the linear y-drift
    # (caused by asymmetric bond angles) so the chain stays near the x-axis.
    n_bb = 3 * n_residues
    bb = np.zeros((n_bb, 3))
    bb[0] = [0., 0., 0.]       # N1
    bb[1] = [r_NCa, 0., 0.]    # Cα1 along +x

    direction = np.array([1.0, 0.0])  # current 2D direction (xy)

    for i in range(2, n_bb):
        vertex = (i - 1) % 3  # 0=N, 1=Cα, 2=C'
        turn_deg = 180.0 - bb_angles[vertex]
        side = 1.0 if (i % 2 == 0) else -1.0
        turn = side * np.radians(turn_deg)
        cos_t, sin_t = np.cos(turn), np.sin(turn)
        direction = np.array([cos_t * direction[0] - sin_t * direction[1],
                              sin_t * direction[0] + cos_t * direction[1]])
        direction /= np.linalg.norm(direction)  # numerical safety
        bond = bb_bonds[(i - 1) % 3]
        bb[i, 0] = bb[i - 1, 0] + bond * direction[0]
        bb[i, 1] = bb[i - 1, 1] + bond * direction[1]

    # Align chain along x-axis by rotating (preserves ALL bond lengths/angles).
    # The zigzag has a net y-drift; rotating by the chain's overall angle
    # brings the drift into x, leaving only the oscillation in y.
    if n_bb > 2:
        dx = bb[-1, 0] - bb[0, 0]
        dy = bb[-1, 1] - bb[0, 1]
        theta = np.arctan2(dy, dx)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_new = bb[:, 0] * cos_t + bb[:, 1] * sin_t
        y_new = -bb[:, 0] * sin_t + bb[:, 1] * cos_t
        bb[:, 0] = x_new
        bb[:, 1] = y_new

    def bb_atom(res, offset):
        return bb[3 * res + offset]

    # ---- side atoms (vector geometry: robust, overlap-free) ----
    def _unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else v

    a_CaCO = 121.0  # not used directly; O placed by bisector method
    atoms = []
    for i in range(n_residues):
        Ni, Cai, Cpi = bb_atom(i, 0), bb_atom(i, 1), bb_atom(i, 2)

        atoms.append(("N", Ni))
        atoms.append(("C", Cai))
        atoms.append(("C", Cpi))

        # Carbonyl O: opposite to bisector of Cα and N(next) from C'
        if i < n_residues - 1:
            N_next = bb_atom(i + 1, 0)
            bis = _unit(_unit(Cai - Cpi) + _unit(N_next - Cpi))
        else:
            bis = _unit(Cai - Cpi)
        O_i = Cpi + r_CO * (-bis)
        atoms.append(("O", O_i))

        # Amide H on N
        if i > 0:
            Cpi_prev = bb_atom(i - 1, 2)
            bis_N = _unit(_unit(Cai - Ni) + _unit(Cpi_prev - Ni))
            HN = Ni + r_NH * (-bis_N)
            atoms.append(("H", HN))
        else:
            axis = _unit(Cai - Ni)
            perp = np.cross(axis, np.array([0., 0., 1.]))
            if np.linalg.norm(perp) < 0.1:
                perp = np.cross(axis, np.array([0., 1., 0.]))
            perp = _unit(perp)
            HN1 = Ni + r_NH * (-0.5 * axis + 0.866 * perp)
            HN2 = Ni + r_NH * (-0.5 * axis - 0.866 * perp)
            atoms.append(("H", HN1))
            atoms.append(("H", HN2))

        # Alpha H's: perpendicular to backbone plane (±z)
        normal = np.cross(Cai - Ni, Cpi - Cai)
        normal = _unit(normal) if np.linalg.norm(normal) > 1e-10 else np.array([0., 1., 0.])
        Ha1 = Cai + r_CaH * normal
        Ha2 = Cai - r_CaH * normal
        atoms.append(("H", Ha1))
        atoms.append(("H", Ha2))

        # C-terminal OH
        if i == n_residues - 1:
            H_OH = O_i + r_OH * _unit(O_i - Cpi)
            atoms.append(("H", H_OH))

    elements = np.array([a[0] for a in atoms])
    coords = np.array([a[1] for a in atoms])
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    coords -= (lo + hi) / 2.0
    return elements, coords
