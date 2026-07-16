"""Cellulose chain generator: beta-1,4-linked D-glucose polymer -(C6H10O5)n-.

Cellulose is the most abundant biopolymer: a sigma-rich polysaccharide
(C-O-C glycosidic + C-C + C-O ring bonds, no conjugated pi), so it is an
insulator with a large (~2.3 eV in LDA/SZ) gap that does NOT shrink with
length. NTPoly/TRS2 therefore converges at any scale, like PE/PEG and unlike
peptide backbones (~0.1 eV amide band gap). MFCC cuts the C-O glycosidic bonds.

Geometry is built by tiling: a real cellotetraose structure (PubChem 3D) is
decomposed into glucose units + glycosidic linkages; the rigid transform
between two internal glucose units is extracted by Kabsch alignment and
applied repeatedly to build an n-glucose chain with correct beta-1,4 geometry.
"""
from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import numpy as np

_TEMPLATE_SDF = Path(__file__).resolve().parent / "cellotetraose.sdf"


def _parse_sdf(path):
    L = Path(path).read_text().split("\n")
    na, nb = (int(x) for x in L[3].split()[:2])
    els = [L[4 + i].split()[3] for i in range(na)]
    co = np.array([[float(x) for x in L[4 + i].split()[:3]] for i in range(na)])
    bonds = []
    for k in range(nb):
        p = L[4 + na + k].split()
        bonds.append((int(p[0]) - 1, int(p[1]) - 1))
    return els, co, bonds


def _adjacency(na, bonds):
    adj = defaultdict(list)
    for i, j in bonds:
        adj[i].append(j)
        adj[j].append(i)
    return adj


def _decompose(els, co, bonds):
    """Return (glucose_units, glycosidic_O_atoms). A glycosidic O is an O bonded
    to 2 C's whose removal disconnects the molecule into large pieces."""
    na = len(els)
    adj = _adjacency(na, bonds)

    def components(removed):
        vis = {removed}
        comps = []
        for s in range(na):
            if s in vis:
                continue
            st = [s]; vis.add(s); c = []
            while st:
                x = st.pop(); c.append(x)
                for y in adj[x]:
                    if y in vis:
                        continue
                    vis.add(y); st.append(y)
            comps.append(c)
        return comps

    glyc_O = []
    for i, e in enumerate(els):
        if e == "O" and len(adj[i]) == 2 and all(els[j] == "C" for j in adj[i]):
            cp = components(i)
            if len(cp) >= 2 and all(len(p) > 5 for p in cp):
                glyc_O.append(i)
    removed = set(glyc_O)
    vis = set(removed); units = []
    for s in range(na):
        if s in vis:
            continue
        st = [s]; vis.add(s); c = []
        while st:
            x = st.pop(); c.append(x)
            for y in adj[x]:
                if y in vis:
                    continue
                vis.add(y); st.append(y)
        units.append(sorted(c))
    return units, glyc_O, adj


def _kabsch(P, Q):
    """Rigid transform (R, t) mapping P onto Q (least-squares). R@P + t ~= Q."""
    cP, cQ = P.mean(0), Q.mean(0)
    H = (P - cP).T @ (Q - cQ)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    t = cQ - R @ cP
    return R, t


def _unit_landmarks(unit_atoms, els, adj):
    """Identify C1, C4, O5 (local indices within the unit) by connectivity.
    Uses the unit-internal graph (glycosidic O already removed, so C1/C4 have
    degree 3 = dangling, other ring C's degree 4). Returns dict local_idx->global."""
    gset = set(unit_atoms)
    g2l = {g: l for l, g in enumerate(unit_atoms)}
    # internal degree = #neighbors that are in the unit
    deg = {}
    nbrs = {}
    for g in unit_atoms:
        ns = [j for j in adj[g] if j in gset]
        deg[g] = len(ns)
        nbrs[g] = ns
    # O5: O with degree 2 and both neighbors C (ring oxygen); OH oxygens have a H neighbor
    O5 = None
    for g in unit_atoms:
        if els[g] == "O" and deg[g] == 2 and all(els[j] == "C" for j in nbrs[g]):
            O5 = g; break
    # C1: degree-3 C with an O neighbor (the anomeric C, bonded to ring O5)
    # C4: degree-3 C with no O neighbor
    deg3C = [g for g in unit_atoms if els[g] == "C" and deg[g] == 3]
    C1 = next((g for g in deg3C if any(els[j] == "O" for j in nbrs[g])), None)
    C4 = next((g for g in deg3C if not any(els[j] == "O" for j in nbrs[g])), None)
    return {"C1": C1, "C4": C4, "O5": O5, "nbrs": nbrs}


def _build_tiler():
    els, co, bonds = _parse_sdf(_TEMPLATE_SDF)
    units, glyc_O, adj = _decompose(els, co, bonds)

    def heavy(u):
        return [a for a in u if els[a] != "H"]
    internal = [u for u in units if len(heavy(u)) == 10]
    if len(internal) < 2:
        raise RuntimeError("need >=2 internal glucose units in cellotetraose template")
    A, B = internal[0], internal[1]
    lmA = _unit_landmarks(A, els, adj)
    lmB = _unit_landmarks(B, els, adj)
    # landmark-based Kabsch: 3 non-collinear points (C1, C4, O5) define the frame
    P = np.array([co[lmA["C1"]], co[lmA["C4"]], co[lmA["O5"]]])
    Q = np.array([co[lmB["C1"]], co[lmB["C4"]], co[lmB["O5"]]])
    R, t = _kabsch(P, Q)
    # forward glycosidic O = the linkage O between A and B; C4 in A is its carbon
    a_set = set(A); b_set = set(B)
    fwd_O = None
    for o in glyc_O:
        cs = [j for j in adj[o] if els[j] == "C"]
        if any(c in a_set for c in cs) and any(c in b_set for c in cs):
            fwd_O = o
    if fwd_O is None:
        raise RuntimeError("could not find A<->B glycosidic linkage")
    return {
        "els": [els[i] for i in A], "co": co[A].copy(),
        "fwd_O": co[fwd_O],
        "C1_local": A.index(lmA["C1"]), "C4_local": A.index(lmA["C4"]),
        "C1_bonded": [A.index(j) for j in lmA["nbrs"][lmA["C1"]]],   # for cap direction
        "C4_bonded": [A.index(j) for j in lmA["nbrs"][lmA["C4"]]],
        "R": R, "t": t,
    }


_TILER = None


def _tiler():
    global _TILER
    if _TILER is None:
        _TILER = _build_tiler()
    return _TILER


def generate_cellulose(n_glucose: int):
    """Build an n_glucose beta-1,4 cellulose chain. Returns (elements, coords).

    Tiles the internal-glucose template by the extracted (R, t) step; inserts a
    glycosidic O between consecutive glucoses; caps the two chain ends with H
    (saturating C1/C4 valences). PCA-rotated for mesh."""
    T = _tiler()
    R, t = T["R"], T["t"]
    g_els = T["els"]; g_co = T["co"]
    fwdO = T["fwd_O"]
    c1l, c4l = T["C1_local"], T["C4_local"]

    # tile: glucose_k = (R,t)^k applied to template; glycosidic O_k likewise.
    gk = g_co.copy(); ok = fwdO.copy()
    glucose_cos = []; o_cos = []
    for k in range(n_glucose):
        glucose_cos.append(gk.copy())
        if k < n_glucose - 1:
            o_cos.append(ok.copy())
        gk = (R @ gk.T).T + t
        ok = R @ ok + t

    atoms = []
    for gc in glucose_cos:
        for e, p in zip(g_els, gc):
            atoms.append((e, p))
    for op in o_cos:
        atoms.append(("O", op))

    # cap the two chain ends with H, along the dangling-bond direction of C1
    # (first glucose) and C4 (last glucose) -- opposite to their 3 bonded atoms.
    def cap_H(center, bonded):
        d = sum(center - b for b in bonded)
        n = np.linalg.norm(d)
        return center + 1.09 * (d / n if n > 1e-12 else np.array([1., 0, 0.]))
    G0, Gn = glucose_cos[0], glucose_cos[-1]
    atoms.append(("H", cap_H(G0[c1l], [G0[b] for b in T["C1_bonded"]])))
    atoms.append(("H", cap_H(Gn[c4l], [Gn[b] for b in T["C4_bonded"]])))

    el = np.array([a[0] for a in atoms])
    co = np.array([a[1] for a in atoms])
    # PCA-rotate chain axis to x
    cen = co - co.mean(0)
    _, _, Vt = np.linalg.svd(cen, full_matrices=False)
    pr = Vt[0]; tg = np.array([1., 0, 0.])
    v = np.cross(pr, tg); s = np.linalg.norm(v)
    if s > 1e-10:
        c = np.dot(pr, tg)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        Mr = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
        co = (Mr @ co.T).T
    lo, hi = co.min(0), co.max(0)
    co -= (lo + hi) / 2
    return el, co
