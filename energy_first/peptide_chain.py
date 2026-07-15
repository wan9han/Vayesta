"""Polyglycine chain generator (simplest peptide) for MFCC generality validation.

Builds a beta-strand polyglycine (-(NH-CH2-CO)-) backbone using NeRF, then
rotates the entire chain to align its principal (longest) axis with x.
The rotation is rigid (preserves ALL bond lengths/angles) and reduces the
y-extent from ~1500 Å to ~5 Å, keeping the SIESTA mesh manageable.

The beta-strand dihedrals (phi=-120, psi=120, omega=180) are physically
realistic and maintain a reasonable HOMO-LUMO gap (peptide planes are
twisted, NOT fully conjugated). The earlier planar zigzag (phi=psi=180)
created full pi-conjugation → near-zero gap → SCF divergence for long chains.
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
    """Build polyglycine C/H/N/O chain (beta-strand, rotated to minimise bbox).

    Returns (elements, coords) numpy arrays centered to bbox center.
    """
    # Bond lengths
    r_NCa, r_CaC, r_CN = 1.46, 1.52, 1.33
    r_CO, r_NH, r_CaH, r_OH = 1.23, 1.01, 1.09, 0.96
    # Bond angles (rad)
    a_NCaC, a_CaCN, a_CNCa = 111*_DEG, 116*_DEG, 123*_DEG
    a_CaCO = 121*_DEG
    a_CaNH = 120*_DEG
    a_CNH  = 120*_DEG
    a_NCaH = 109.5*_DEG
    a_COH  = 109.5*_DEG
    # Beta-strand dihedrals: physically realistic, NOT fully conjugated
    phi_v, psi_v, omega_v = -120*_DEG, 120*_DEG, 180*_DEG

    # ---- backbone via NeRF ----
    bb = [np.array([0., 0., 0.])]
    bb.append(bb[-1] + r_NCa * np.array([1., 0., 0.]))
    bb.append(bb[-1] + r_CaC * np.array([np.cos(np.pi-a_NCaC), np.sin(np.pi-a_NCaC), 0.]))

    for _ in range(1, n_residues):
        bb.append(_nerf(bb[-3], bb[-2], bb[-1], r_CN,  a_CaCN, psi_v))
        bb.append(_nerf(bb[-3], bb[-2], bb[-1], r_NCa, a_CNCa, omega_v))
        bb.append(_nerf(bb[-3], bb[-2], bb[-1], r_CaC, a_NCaC, phi_v))

    bb_arr = np.array(bb)

    # ---- rotate chain to align principal axis with x ----
    # The beta-strand extends in a 3D direction; PCA finds the longest axis
    # and we rotate it onto x, compressing y/z extent from ~1500 Å to ~5 Å.
    centered = bb_arr - bb_arr.mean(axis=0)
    # SVD: principal components
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    # Vt[0] is the direction of greatest variance = chain axis
    # Rotation matrix: rotate Vt[0] onto [1,0,0]
    principal = Vt[0]  # chain direction (unit vector)
    # Build rotation that maps principal -> [1,0,0]
    target = np.array([1., 0., 0.])
    # Rodrigues rotation or simple Gram-Schmidt
    v = np.cross(principal, target)
    s = np.linalg.norm(v)
    if s > 1e-10:
        c = np.dot(principal, target)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    else:
        R = np.eye(3)
    bb_arr = (R @ bb_arr.T).T

    def bb_atom(res, offset):
        return bb_arr[3 * res + offset]

    # ---- side atoms (vector geometry) ----
    def _unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else v

    atoms = []
    for i in range(n_residues):
        Ni, Cai, Cpi = bb_atom(i, 0), bb_atom(i, 1), bb_atom(i, 2)

        atoms.append(("N", Ni))
        atoms.append(("C", Cai))
        atoms.append(("C", Cpi))

        # Carbonyl O
        if i < n_residues - 1:
            N_next = bb_atom(i + 1, 0)
            bis = _unit(_unit(Cai - Cpi) + _unit(N_next - Cpi))
        else:
            bis = _unit(Cai - Cpi)
        O_i = Cpi + r_CO * (-bis)
        atoms.append(("O", O_i))

        # Amide H
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
            atoms.append(("H", Ni + r_NH * (-0.5 * axis + 0.866 * perp)))
            atoms.append(("H", Ni + r_NH * (-0.5 * axis - 0.866 * perp)))

        # Alpha H's: ±normal to backbone plane
        normal = np.cross(Cai - Ni, Cpi - Cai)
        normal = _unit(normal) if np.linalg.norm(normal) > 1e-10 else np.array([0., 1., 0.])
        atoms.append(("H", Cai + r_CaH * normal))
        atoms.append(("H", Cai - r_CaH * normal))

        # C-terminal OH
        if i == n_residues - 1:
            atoms.append(("H", O_i + r_OH * _unit(O_i - Cpi)))

    elements = np.array([a[0] for a in atoms])
    coords = np.array([a[1] for a in atoms])
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    coords -= (lo + hi) / 2.0
    return elements, coords
