"""Minimal SIESTA backend: run an isolated molecule, return its total energy.

Used by Stage 1 to get ``E_full`` and each capped-fragment / conjugate-cap
energy under identical numerical settings. This is deliberately small and
independent of the old Vayesta adapter.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional

from .molecule import Molecule, write_siesta_fdf

_TOTAL_RE = re.compile(r"^\s*siesta:.*Total\s*=\s*(-?\d+\.\d+)", re.MULTILINE)
_SCF_DONE_RE = re.compile(r"scf:\s*1\s|SCF cycle|Final energy", re.IGNORECASE)


def _clean_mpi_env() -> Dict[str, str]:
    """Drop inherited MPI launcher vars so siesta runs as a clean singleton."""
    env = {k: v for k, v in os.environ.items()}
    for k in list(env):
        if k.startswith(("OMPI_", "PMI_", "PMIX_")):
            env.pop(k, None)
    # Deterministic single-thread BLAS for these tiny systems.
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    return env


def run_siesta(
    mol: Molecule,
    workdir,
    siesta_bin: str,
    pseudo_dir: str,
    label: Optional[str] = None,
    cell_margin: float = 8.0,
    timeout: float = 600.0,
    basis_size: str = "SZ",
    mesh_cutoff_ry: float = 100.0,
) -> Dict[str, object]:
    """Run SIESTA on ``mol``; return ``{energy_ev, converged, returncode, ...}``.

    ``workdir`` is created (or reused). Pseudopotentials are copied from
    ``pseudo_dir`` for every element present. The FDF is written with an
    explicit vacuum cell so the result is comparable across molecules.
    ``basis_size`` / ``mesh_cutoff_ry`` select the numerical level (identical
    for full + fragments + caps to keep MFCC comparisons bias-free).
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    label = label or mol.label or "mol"
    mol.label = label

    fdf_path = workdir / f"{label}.fdf"
    write_siesta_fdf(
        mol, fdf_path, cell_margin=cell_margin,
        basis_size=basis_size, mesh_cutoff_ry=mesh_cutoff_ry,
    )

    # Copy pseudopotentials for the species we actually use.
    for el in set(mol.elements):
        src = Path(pseudo_dir) / f"{el}.psf"
        if not src.exists():
            raise FileNotFoundError(f"pseudo {src} not found")
        shutil.copy2(src, workdir / f"{el}.psf")

    env = _clean_mpi_env()
    with open(fdf_path) as fh:
        proc = subprocess.run(
            [siesta_bin],
            stdin=fh,
            cwd=workdir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            text=True,
        )
    out_path = workdir / "siesta.out"
    out_path.write_text(proc.stdout)

    energy = None
    m = None
    for m in _TOTAL_RE.finditer(proc.stdout):
        last = m
    if m is not None:
        energy = float(last.group(1))

    converged = energy is not None and "SCF convergence" in proc.stdout or energy is not None
    # A present, converged energy == a final "Total =" line was printed.
    converged = energy is not None
    return {
        "label": label,
        "energy_ev": energy,
        "converged": converged,
        "returncode": proc.returncode,
        "workdir": str(workdir),
        "natoms": mol.natoms,
        "formula": _formula(mol),
    }


def _formula(mol: Molecule) -> str:
    counts: Dict[str, int] = {}
    for el in mol.elements:
        counts[el] = counts.get(el, 0) + 1
    return "".join(f"{el}{counts[el]}" for el in sorted(counts, key=lambda e: _AN(e)))


def _AN(el: str) -> int:
    return {"H": 1, "C": 6}.get(el, 99)
