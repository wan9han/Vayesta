"""Polyglycine chain generator (simplest peptide) for MFCC generality validation.

Builds an alpha-helix polyglycine (-(NH-CH2-CO)-) backbone using NeRF, then
rotates the entire chain to align its principal (longest) axis with x.
The rotation is rigid (preserves ALL bond lengths/angles) and reduces the
y/z-extent to a few Å, keeping the SIESTA mesh manageable.

WHY ALPHA-HELIX (not beta-strand): a regular peptide backbone is a 1D polymer
with a HOMO-LUMO gap that SHRINKS with chain length. The extended beta-strand
(phi=-120, psi=120) collapses to ~0.1 eV at 64 residues -- too small for the
NTPoly/TRS2 density-matrix purification to resolve, so SCF oscillates forever.
The more compact alpha-helix fold (phi=-57, psi=-47) localises the amide
states, keeping the gap ~4x larger at the same length, so TRS2 converges.
(For TRS2 SCF one must also set SCF.H.Converge .false. in molecule.py: the
borderline orbital near the small gap makes dHmax oscillate even though the
density matrix and total energy are converged.)
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
    """Build ACE/NME-capped alpha-helix polyglycine: Ac-(Gly)_n-NHMe.

    Returns (elements, coords) numpy arrays (PCA-rotated, centered to bbox).

    Two choices that let NTPoly/TRS2 converge:
      * alpha-helix dihedrals (phi=-57, psi=-47) -- stable ~1 eV bulk gap.
      * ACE/NME end caps -- remove the free -NH2/-COOH termini whose
        charge-transfer state pins the bare-chain gap at ~0.06 eV (far too
        small for density-matrix purification). Capping opens it to ~1 eV.
    """
    # Bond lengths / angles
    r_NCa, r_CaC, r_CN = 1.46, 1.52, 1.33
    r_CO, r_NH, r_CaH, r_CC, r_CH = 1.23, 1.01, 1.09, 1.52, 1.09
    a_NCaC, a_CaCN, a_CNCa = 111*_DEG, 116*_DEG, 123*_DEG
    # Alpha-helix dihedrals
    phi_v, psi_v, omega_v = -57*_DEG, -47*_DEG, 180*_DEG

    def _unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else v

    def _methyl(C_ref, parent):
        ax = _unit(C_ref - parent)
        t1 = _unit(np.cross(ax, np.array([0., 0., 1.])))
        if np.linalg.norm(t1) < 0.1:
            t1 = _unit(np.cross(ax, np.array([0., 1., 0.])))
        t2 = np.cross(ax, t1)
        out = [("C", C_ref)]
        for th in (0., 2*np.pi/3, 4*np.pi/3):
            d = (-1/3.)*ax + (2*np.sqrt(2)/3.)*(np.cos(th)*t1 + np.sin(th)*t2)
            out.append(("H", C_ref + r_CH*d))
        return out

    def _trigonal_two(C_center, d_known, r1, r2, plane_normal):
        d = _unit(d_known)
        p = _unit(np.cross(plane_normal, d))
        d1 = np.cos(120*_DEG)*d + np.sin(120*_DEG)*p
        d2 = np.cos(120*_DEG)*d - np.sin(120*_DEG)*p
        return C_center + r1*d1, C_center + r2*d2

    # ---- backbone via NeRF ----
    bb = [np.array([0., 0., 0.])]
    bb.append(bb[-1] + r_NCa * np.array([1., 0., 0.]))
    bb.append(bb[-1] + r_CaC * np.array([np.cos(np.pi-a_NCaC), np.sin(np.pi-a_NCaC), 0.]))
    for _ in range(1, n_residues):
        bb.append(_nerf(bb[-3], bb[-2], bb[-1], r_CN,  a_CaCN, psi_v))
        bb.append(_nerf(bb[-3], bb[-2], bb[-1], r_NCa, a_CNCa, omega_v))
        bb.append(_nerf(bb[-3], bb[-2], bb[-1], r_CaC, a_NCaC, phi_v))
    bb = np.array(bb)
    def b(i, o): return bb[3*i + o]

    # ---- ACE cap (trans amide off N0): C_ace(=O)-CH3 ----
    N0, Ca0, Cp0 = b(0, 0), b(0, 1), b(0, 2)
    C_ace = _nerf(Cp0, Ca0, N0, r_CN, a_CNCa, 180*_DEG)
    plane_n = _unit(np.cross(N0 - C_ace, Ca0 - N0))
    O_ace, C_me_ace = _trigonal_two(C_ace, N0 - C_ace, r_CO, r_CC, plane_n)
    cap_n = len(bb)  # placeholder; caps collected separately
    ace_atoms = [("C", C_ace), ("O", O_ace)] + _methyl(C_me_ace, C_ace)

    # ---- NME cap (trans amide off C'n): -NH-CH3 ----
    CAn, CPn = b(n_residues - 1, 1), b(n_residues - 1, 2)
    N_nme = _nerf(b(n_residues - 1, 0), CAn, CPn, r_CN, a_CaCN, 120*_DEG)
    plane_n2 = _unit(np.cross(CPn - N_nme, CAn - CPn))
    H_nme, C_me_nme = _trigonal_two(N_nme, CPn - N_nme, r_NH, r_CC, plane_n2)
    nme_atoms = [("N", N_nme), ("H", H_nme)] + _methyl(C_me_nme, N_nme)

    # ---- backbone + side atoms (every residue is now internal-like: N0 is an
    # amide N bonded to C_ace -> one amide H; last C' bonded to N_nme -> normal
    # carbonyl, no OH) ----
    atoms = list(ace_atoms)
    for i in range(n_residues):
        Ni, Cai, Cpi = b(i, 0), b(i, 1), b(i, 2)
        atoms.append(("N", Ni))
        atoms.append(("C", Cai))
        atoms.append(("C", Cpi))
        # carbonyl O (last residue uses the NME nitrogen as "next N")
        N_next = b(i + 1, 0) if i < n_residues - 1 else N_nme
        bis = _unit(_unit(Cai - Cpi) + _unit(N_next - Cpi))
        atoms.append(("O", Cpi + r_CO * (-bis)))
        # amide H (residue 0 uses the ACE carbonyl C as "prev C'")
        Cp_prev = b(i - 1, 2) if i > 0 else C_ace
        bis_N = _unit(_unit(Cai - Ni) + _unit(Cp_prev - Ni))
        atoms.append(("H", Ni + r_NH * (-bis_N)))
        # alpha H's: +/- normal to backbone plane
        normal = np.cross(Cai - Ni, Cpi - Cai)
        normal = _unit(normal) if np.linalg.norm(normal) > 1e-10 else np.array([0., 1., 0.])
        atoms.append(("H", Cai + r_CaH * normal))
        atoms.append(("H", Cai - r_CaH * normal))
    atoms += nme_atoms

    coords = np.array([a[1] for a in atoms])
    elements = np.array([a[0] for a in atoms])

    # ---- PCA-rotate all atoms so the helix axis aligns with x (mesh control:
    # compresses y/z from ~bbox to ~5 A, the helix radius) ----
    centered = coords - coords.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    principal = Vt[0]
    target = np.array([1., 0., 0.])
    v = np.cross(principal, target)
    s = np.linalg.norm(v)
    if s > 1e-10:
        c = np.dot(principal, target)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
        coords = (R @ coords.T).T

    lo, hi = coords.min(axis=0), coords.max(axis=0)
    coords -= (lo + hi) / 2.0
    return elements, coords
