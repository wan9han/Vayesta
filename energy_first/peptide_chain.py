"""Polyglycine chain generator (simplest peptide) for MFCC generality validation.

Builds an extended-chain polyglycine (-(NH-CH2-CO)-) backbone using NeRF
and places side atoms (carbonyl O, amide H, alpha H's, terminal caps) from
internal coordinates. Demonstrates the MFCC+MBE(2) framework works beyond PE.
"""

from __future__ import annotations

import numpy as np

_DEG = np.pi / 180.0


def _nerf(d, c, b, r, theta, phi):
    """Place atom A bonded to B at length r, angle C-B-A=theta, dihedral D-C-B-A=phi."""
    v21 = b - c
    l21 = np.linalg.norm(v21)
    if l21 < 1e-12:
        raise ValueError("coincident atoms B,C")
    d21 = v21 / l21
    v32 = c - d
    n = np.cross(v32, v21)
    ln = np.linalg.norm(n)
    if ln < 1e-12:
        n = np.array([0.0, 0.0, 1.0])
    else:
        n /= ln
    m = np.cross(n, d21)
    new_dir = -np.cos(theta) * d21 + np.sin(theta) * (np.cos(phi) * m + np.sin(phi) * n)
    return b + r * new_dir


def generate_polyglycine(n_residues: int):
    """Build polyglycine C_n H_{...} N_n O_n chain (extended beta-strand).

    Returns (elements, coords) numpy arrays centered to bbox center.
    """
    # Bond lengths
    r_NCa, r_CaC, r_CN = 1.46, 1.52, 1.33
    r_CO, r_NH, r_CaH, r_OH = 1.23, 1.01, 1.09, 0.96
    # Angles (rad)
    a_NCaC, a_CaCN, a_CNCa = 111*_DEG, 116*_DEG, 123*_DEG
    a_CaCO = 121*_DEG
    a_CaNH = 120*_DEG   # Cα-N-H angle
    a_CNH  = 120*_DEG   # C'(prev)-N-H angle
    a_NCaH = 109.5*_DEG
    a_COH  = 109.5*_DEG
    # Extended-chain dihedrals
    phi_v, psi_v, omega_v = -120*_DEG, 120*_DEG, 180*_DEG

    # ---- backbone ----
    bb = [np.array([0., 0., 0.])]                          # N1
    bb.append(bb[-1] + r_NCa * np.array([1., 0., 0.]))     # Cα1
    bb.append(bb[-1] + r_CaC * np.array([np.cos(np.pi-a_NCaC), np.sin(np.pi-a_NCaC), 0.]))  # C'1

    for _ in range(1, n_residues):
        bb.append(_nerf(bb[-3], bb[-2], bb[-1], r_CN,  a_CaCN, psi_v))    # N
        bb.append(_nerf(bb[-3], bb[-2], bb[-1], r_NCa, a_CNCa, omega_v))  # Cα
        bb.append(_nerf(bb[-3], bb[-2], bb[-1], r_CaC, a_NCaC, phi_v))    # C'

    def bb_atom(res, offset):
        return bb[3 * res + offset]  # 0=N, 1=Cα, 2=C'

    # ---- side atoms (vector geometry: robust, overlap-free) ----
    def _unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else v

    atoms = []
    for i in range(n_residues):
        Ni, Cai, Cpi = bb_atom(i, 0), bb_atom(i, 1), bb_atom(i, 2)

        atoms.append(("N", Ni))
        atoms.append(("C", Cai))
        atoms.append(("C", Cpi))

        # Carbonyl O: opposite to the bisector of Cα and N(next) from C'
        if i < n_residues - 1:
            N_next = bb_atom(i + 1, 0)
            bis = _unit(_unit(Cai - Cpi) + _unit(N_next - Cpi))
        else:
            bis = _unit(Cai - Cpi)
        O_i = Cpi + r_CO * (-bis)
        atoms.append(("O", O_i))

        # Amide H on N: opposite to bisector of Cα and C'(prev)
        if i > 0:
            Cpi_prev = bb_atom(i - 1, 2)
            bis_N = _unit(_unit(Cai - Ni) + _unit(Cpi_prev - Ni))
            HN = Ni + r_NH * (-bis_N)
            atoms.append(("H", HN))
        else:
            # N-terminal: 2 H's, symmetric about the N→Cα axis, in the backbone plane
            axis = _unit(Cai - Ni)
            perp = np.cross(axis, np.array([0., 0., 1.]))
            if np.linalg.norm(perp) < 0.1:
                perp = np.cross(axis, np.array([0., 1., 0.]))
            perp = _unit(perp)
            HN1 = Ni + r_NH * (-0.5 * axis + 0.866 * perp)
            HN2 = Ni + r_NH * (-0.5 * axis - 0.866 * perp)
            atoms.append(("H", HN1))
            atoms.append(("H", HN2))

        # Alpha H's: perpendicular to backbone plane (cross product of N→Cα and Cα→C')
        normal = np.cross(Cai - Ni, Cpi - Cai)
        normal = _unit(normal) if np.linalg.norm(normal) > 1e-10 else np.array([0., 1., 0.])
        Ha1 = Cai + r_CaH * normal
        Ha2 = Cai - r_CaH * normal
        atoms.append(("H", Ha1))
        atoms.append(("H", Ha2))

        # C-terminal OH: H bonded to carbonyl O, opposite to C'
        if i == n_residues - 1:
            H_OH = O_i + r_OH * _unit(O_i - Cpi)
            atoms.append(("H", H_OH))

    elements = np.array([a[0] for a in atoms])
    coords = np.array([a[1] for a in atoms])
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    coords -= (lo + hi) / 2.0
    return elements, coords
