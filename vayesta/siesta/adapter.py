"""Minimal SIESTA adapter for coarse contiguous EWF blocks.

This module intentionally keeps the first integration layer independent of the
PySCF-backed EWF internals.  The data structures record the atom and block
mapping that an EWF fragment/backend can later consume directly.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclasses.dataclass(frozen=True)
class Atom:
    """One SIESTA coordinate entry with its global atom index."""

    global_index: int
    x: float
    y: float
    z: float
    species: int


@dataclasses.dataclass(frozen=True)
class FdfInput:
    """Parsed subset of an FDF file needed to emit local block inputs."""

    lines: tuple[str, ...]
    atoms: tuple[Atom, ...]
    coordinates_start: int
    coordinates_end: int
    species_labels: dict[int, str] = dataclasses.field(default_factory=dict)
    species_atomic_numbers: dict[int, int] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class SiestaBlock:
    """Contiguous EWF block with core atoms and input atoms including buffer."""

    block_id: int
    core_atom_start: int
    core_atom_end: int
    input_atom_start: int
    input_atom_end: int
    machine_id: int | None = None

    @property
    def buffer_left(self) -> int:
        return self.core_atom_start - self.input_atom_start

    @property
    def buffer_right(self) -> int:
        return self.input_atom_end - self.core_atom_end

    def to_metadata(self, local_to_global: Sequence[int]) -> dict:
        return {
            "block_id": self.block_id,
            "machine_id": self.machine_id,
            "core_atom_start": self.core_atom_start,
            "core_atom_end": self.core_atom_end,
            "input_atom_start": self.input_atom_start,
            "input_atom_end": self.input_atom_end,
            "buffer_left": self.buffer_left,
            "buffer_right": self.buffer_right,
            "local_to_global_atom_index": list(local_to_global),
        }


@dataclasses.dataclass(frozen=True)
class SiestaSolverConfig:
    """SIESTA/ELSI solver settings forced for each local block."""

    ntpoly_method: int = 2
    ntpoly_filter: float = 1.0e-9
    ntpoly_tolerance: float = 1.0e-6
    max_scf_iterations: int = 150
    dm_number_pulay: int = 6
    dm_mixing_weight: float = 0.05

    def to_fdf_options(self) -> dict[str, str]:
        return {
            "solutionmethod": "SolutionMethod     ELSI",
            "elsisolver": "ELSI.Solver        ntpoly",
            "elsintpolymethod": f"ELSI.NTPoly.Method {self.ntpoly_method}",
            "elsintpolyfilter": f"ELSI.NTPoly.Filter {self.ntpoly_filter:.1e}",
            "elsintpolytolerance": f"ELSI.NTPoly.Tolerance {self.ntpoly_tolerance:.1e}",
            "maxscfiterations": f"MaxSCFIterations    {self.max_scf_iterations}",
            "dmnumberpulay": f"DM.NumberPulay    {self.dm_number_pulay}",
            "dmmixingweight": f"DM.MixingWeight    {self.dm_mixing_weight:.6f}",
            "writedm": "WriteDM          true",
            "savehs": "SaveHS           true",
            "writeorbitalindex": "WriteOrbitalIndex true",
        }

    def to_metadata(self) -> dict:
        return {
            "solution_method": "ELSI",
            "elsi_solver": "ntpoly",
            "ntpoly_method": self.ntpoly_method,
            "ntpoly_filter": self.ntpoly_filter,
            "ntpoly_tolerance": self.ntpoly_tolerance,
            "max_scf_iterations": self.max_scf_iterations,
            "dm_number_pulay": self.dm_number_pulay,
            "dm_mixing_weight": self.dm_mixing_weight,
        }


@dataclasses.dataclass(frozen=True)
class SiestaRunConfig:
    """Environment-controlled SIESTA adapter configuration."""

    num_machines: int
    procs_per_machine: int
    threads_per_proc: int
    workdir: Path
    siesta_bin: str | None
    block_atoms: int | None
    buffer_atoms: int
    block_groups: int | None
    group_size_atoms: int | None
    buffer_groups: int
    terminal_cap_atoms: int
    dry_run: bool
    predictive_boundary: bool = False
    predictive_boundary_damping: float = 1.0
    predictive_boundary_rerun: bool = False
    effective_interaction_u_ev: float = 0.0
    effective_interaction_denominator_shift_ev: float = 1.0e-6
    solver: SiestaSolverConfig = dataclasses.field(default_factory=SiestaSolverConfig)


@dataclasses.dataclass(frozen=True)
class SiestaResult:
    """Minimal data returned from a local SIESTA block calculation."""

    block_id: int
    converged: bool | None
    total_energy_ev: float | None
    wall_time_seconds: float | None
    density_matrix_path: Path | None
    hamiltonian_matrix_path: Path | None
    overlap_matrix_path: Path | None
    orbital_index_path: Path | None
    atom_orbital_ranges: dict[int, tuple[int, int]]
    output_path: Path
    matrix_metadata: dict[str, dict] = dataclasses.field(default_factory=dict)
    run_diagnostics: dict[str, object] = dataclasses.field(default_factory=dict)

    def to_metadata(self) -> dict:
        return {
            "block_id": self.block_id,
            "converged": self.converged,
            "total_energy_ev": self.total_energy_ev,
            "wall_time_seconds": self.wall_time_seconds,
            "density_matrix_path": _path_to_str(self.density_matrix_path),
            "hamiltonian_matrix_path": _path_to_str(self.hamiltonian_matrix_path),
            "overlap_matrix_path": _path_to_str(self.overlap_matrix_path),
            "orbital_index_path": _path_to_str(self.orbital_index_path),
            "atom_orbital_ranges": {
                str(atom): [start, end] for atom, (start, end) in sorted(self.atom_orbital_ranges.items())
            },
            "output_path": str(self.output_path),
            "matrix_metadata": self.matrix_metadata,
            "run_diagnostics": self.run_diagnostics,
        }


@dataclasses.dataclass(frozen=True)
class SiestaMatrixMetadata:
    """Header-level metadata for a SIESTA matrix artifact."""

    kind: str
    path: Path
    norbitals: int
    nspin: int
    nsc: tuple[int, int, int] | None = None
    natoms: int | None = None
    nspecies: int | None = None
    version: int | None = None
    double_precision: bool | None = None

    def to_metadata(self) -> dict:
        return {
            "kind": self.kind,
            "path": str(self.path),
            "norbitals": self.norbitals,
            "nspin": self.nspin,
            "nsc": None if self.nsc is None else list(self.nsc),
            "natoms": self.natoms,
            "nspecies": self.nspecies,
            "version": self.version,
            "double_precision": self.double_precision,
        }


@dataclasses.dataclass(frozen=True)
class SiestaHsxMatrix:
    """Sparse Hamiltonian/overlap data from a SIESTA `.HSX` file."""

    metadata: SiestaMatrixMetadata
    rows: np.ndarray
    cols: np.ndarray
    hamiltonian: np.ndarray
    overlap: np.ndarray

    @property
    def nnz(self) -> int:
        return int(self.rows.size)

    def core_block(self, orbital_ranges: dict[int, tuple[int, int]]) -> "SiestaHsxCoreBlock":
        indices = sorted(
            orbital
            for start, end in orbital_ranges.values()
            for orbital in range(start, end)
        )
        mask = np.isin(self.rows, indices) & np.isin(self.cols, indices)
        return SiestaHsxCoreBlock(
            orbital_indices=np.asarray(indices, dtype=np.int64),
            rows=self.rows[mask],
            cols=self.cols[mask],
            hamiltonian=self.hamiltonian[:, mask],
            overlap=self.overlap[mask],
        )


@dataclasses.dataclass(frozen=True)
class SiestaHsxCoreBlock:
    """Core-orbital subblock view of a sparse HSX matrix."""

    orbital_indices: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    hamiltonian: np.ndarray
    overlap: np.ndarray

    @property
    def nnz(self) -> int:
        return int(self.rows.size)

    def to_metadata(self) -> dict:
        return {
            "norbitals": int(self.orbital_indices.size),
            "nnz": self.nnz,
            "orbital_start": None if self.orbital_indices.size == 0 else int(self.orbital_indices[0]),
            "orbital_end": None if self.orbital_indices.size == 0 else int(self.orbital_indices[-1] + 1),
        }


@dataclasses.dataclass(frozen=True)
class SiestaDensityMatrix:
    """Sparse density matrix data from a SIESTA `.DM` file."""

    metadata: SiestaMatrixMetadata
    rows: np.ndarray
    cols: np.ndarray
    density: np.ndarray

    @property
    def nnz(self) -> int:
        return int(self.rows.size)

    def core_block(self, orbital_ranges: dict[int, tuple[int, int]]) -> "SiestaSparseCoreBlock":
        indices = sorted(
            orbital
            for start, end in orbital_ranges.values()
            for orbital in range(start, end)
        )
        mask = np.isin(self.rows, indices) & np.isin(self.cols, indices)
        return SiestaSparseCoreBlock(
            kind="density",
            orbital_indices=np.asarray(indices, dtype=np.int64),
            rows=self.rows[mask],
            cols=self.cols[mask],
            values=self.density[:, mask],
        )


@dataclasses.dataclass(frozen=True)
class SiestaSparseCoreBlock:
    """Core-orbital sparse subblock summary for a SIESTA matrix."""

    kind: str
    orbital_indices: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    values: np.ndarray

    @property
    def nnz(self) -> int:
        return int(self.rows.size)

    def to_metadata(self) -> dict:
        return {
            "kind": self.kind,
            "norbitals": int(self.orbital_indices.size),
            "nnz": self.nnz,
            "orbital_start": None if self.orbital_indices.size == 0 else int(self.orbital_indices[0]),
            "orbital_end": None if self.orbital_indices.size == 0 else int(self.orbital_indices[-1] + 1),
        }


@dataclasses.dataclass(frozen=True)
class SiestaGlobalMatrices:
    """Core-owned global sparse matrices assembled from SIESTA block results."""

    atom_orbital_ranges: dict[int, tuple[int, int]]
    rows: np.ndarray
    cols: np.ndarray
    density: np.ndarray
    hamiltonian: np.ndarray
    overlap: np.ndarray
    block_ids: np.ndarray

    @property
    def norbitals(self) -> int:
        if not self.atom_orbital_ranges:
            return 0
        return max(end for _, end in self.atom_orbital_ranges.values())

    @property
    def nnz(self) -> int:
        return int(self.rows.size)

    @property
    def density_overlap_trace_by_spin(self) -> tuple[float, ...]:
        if self.density.size == 0 or self.overlap.size == 0:
            return ()
        return tuple(float(np.dot(spin_density, self.overlap)) for spin_density in self.density)

    @property
    def density_overlap_trace_total(self) -> float | None:
        by_spin = self.density_overlap_trace_by_spin
        if not by_spin:
            return None
        return float(sum(by_spin))

    def to_metadata(self) -> dict:
        return {
            "natoms": len(self.atom_orbital_ranges),
            "norbitals": self.norbitals,
            "nnz": self.nnz,
            "nspin_density": int(self.density.shape[0]) if self.density.size else 0,
            "nspin_hamiltonian": int(self.hamiltonian.shape[0]) if self.hamiltonian.size else 0,
            "density_overlap_trace_by_spin": list(self.density_overlap_trace_by_spin),
            "density_overlap_trace_total": self.density_overlap_trace_total,
            "atom_orbital_ranges": {
                str(atom): [start, end]
                for atom, (start, end) in sorted(self.atom_orbital_ranges.items())
            },
        }


@dataclasses.dataclass(frozen=True)
class SiestaValidationReport:
    """Machine-readable diagnostics for the current core-owned approximation."""

    ok: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    natoms: int | None
    nblocks: int
    ncore_atoms: int
    norbitals: int
    nnz: int
    density_overlap_trace_total: float | None
    density_overlap_trace_by_spin: tuple[float, ...]
    total_block_energy_ev: float | None
    energy_policy: str

    def to_metadata(self) -> dict:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "natoms": self.natoms,
            "nblocks": self.nblocks,
            "ncore_atoms": self.ncore_atoms,
            "norbitals": self.norbitals,
            "nnz": self.nnz,
            "density_overlap_trace_total": self.density_overlap_trace_total,
            "density_overlap_trace_by_spin": list(self.density_overlap_trace_by_spin),
            "total_block_energy_ev": self.total_block_energy_ev,
            "energy_policy": self.energy_policy,
        }


@dataclasses.dataclass(frozen=True)
class SiestaBlockWorkflow:
    """Reusable EWF -> SIESTA -> EWF block workflow facade."""

    fdf: FdfInput
    config: SiestaRunConfig
    blocks: tuple[SiestaBlock, ...]
    block_dirs: tuple[Path, ...]
    pseudopotentials: tuple[Path, ...] = ()

    @property
    def natoms(self) -> int:
        return len(self.fdf.atoms)

    def write_inputs(self) -> list[Path]:
        block_dirs = generate_block_directories(
            self.fdf,
            self.blocks,
            self.config.workdir,
            pseudopotentials=self.pseudopotentials,
            siesta_bin=self.config.siesta_bin,
            threads_per_proc=self.config.threads_per_proc,
            solver_config=self.config.solver,
        )
        write_block_manifest(self.config.workdir, block_dirs)
        write_schedule_manifest(
            self.config.workdir,
            self.blocks,
            num_machines=self.config.num_machines,
            procs_per_machine=self.config.procs_per_machine,
            threads_per_proc=self.config.threads_per_proc,
        )
        write_boundary_manifest(self.config.workdir, self.fdf, self.blocks)
        write_embedding_contract_manifest(self.config.workdir)
        write_boundary_corrections_manifest(self.config.workdir)
        return block_dirs

    def assigned_blocks(self, machine_id: int, local_rank: int) -> list[SiestaBlock]:
        return assign_blocks_to_rank(
            self.blocks,
            machine_id=machine_id,
            local_rank=local_rank,
            procs_per_machine=self.config.procs_per_machine,
        )

    def assigned_dirs(self, machine_id: int, local_rank: int) -> list[Path]:
        return [self.block_dirs[block.block_id] for block in self.assigned_blocks(machine_id, local_rank)]

    def run_rank(self, rank: int, machine_id: int, local_rank: int) -> tuple[list[SiestaResult], list[subprocess.CompletedProcess]]:
        assigned_dirs = self.assigned_dirs(machine_id, local_rank)
        completed = run_assigned_blocks(assigned_dirs, self.config)
        parsed = [read_siesta_output(block_dir) for block_dir in assigned_dirs[: len(completed)]]
        write_rank_results(self.config.workdir, rank, parsed, completed)
        return parsed, completed

    def finalize(self) -> dict:
        payload = {
            "results": write_results_manifest(self.config.workdir),
            "run_summary": write_run_summary_manifest(self.config.workdir),
            "matrix_shape_report": None,
            "weak_scaling_report": None,
            "validation": None,
            "ewf_results": None,
            "global_matrices": None,
            "electron_constraint": None,
            "embedded_observables": None,
            "physical_readiness": None,
        }
        payload["matrix_shape_report"] = write_matrix_shape_report_manifest(self.config.workdir)
        payload["weak_scaling_report"] = write_weak_scaling_report(
            self.config.workdir / "weak_scaling_report.json",
            [self.config.workdir],
        )
        if self.config.dry_run:
            return payload
        validation = write_validation_manifest(
            self.config.workdir,
            natoms=self.natoms,
            min_buffer_atoms=minimum_buffer_atoms(self.config),
        )
        payload["validation"] = validation
        if validation["ok"]:
            payload["ewf_results"] = write_ewf_results_manifest(self.config.workdir)
            payload["global_matrices"] = write_global_matrices_manifest(self.config.workdir, natoms=self.natoms)
            payload["matrix_shape_report"] = write_matrix_shape_report_manifest(self.config.workdir)
            payload["electron_constraint"] = write_electron_constraint_manifest(
                self.config.workdir,
                self.fdf,
            )
            if self.config.predictive_boundary:
                payload["predictive_embedding_potential"] = write_predictive_boundary_potential_manifest(
                    self.config.workdir,
                    damping=self.config.predictive_boundary_damping,
                )
                payload["boundary_corrections"] = write_predictive_boundary_corrections_manifest(
                    self.config.workdir,
                    damping=self.config.predictive_boundary_damping,
                )
            payload["embedded_observables"] = write_embedded_observables_manifest(self.config.workdir)
            if self.config.predictive_boundary:
                payload["predictive_ewf_closure"] = write_predictive_ewf_closure_manifest(self.config.workdir)
                payload["cluster_hamiltonians"] = write_cluster_hamiltonians_manifest(self.config.workdir)
                payload["cluster_solver_results"] = write_cluster_solver_results_manifest(self.config.workdir)
                payload["effective_correlated_results"] = write_effective_correlated_results_manifest(
                    self.config.workdir,
                    effective_interaction_u_ev=self.config.effective_interaction_u_ev,
                    denominator_shift_ev=self.config.effective_interaction_denominator_shift_ev,
                )
                payload["embedded_observables"] = write_embedded_observables_manifest(self.config.workdir)
            payload["validation"] = write_validation_manifest(
                self.config.workdir,
                natoms=self.natoms,
                min_buffer_atoms=minimum_buffer_atoms(self.config),
            )
        if (self.config.workdir / "validation.json").exists():
            payload["physical_readiness"] = write_physical_readiness_manifest(self.config.workdir)
        return payload

    def write_predictive_embedding_inputs(self) -> dict:
        """Write SIESTA-readable embedding potentials for a predictive rerun."""

        return write_siesta_embedding_potential_inputs(self.config.workdir)

    def snapshot_results(self, label: str) -> dict:
        """Save the current rank/result manifests under a phase label."""

        return write_results_snapshot_manifest(self.config.workdir, label)

    def write_embedding_rerun_delta(self) -> dict:
        """Write first-pass versus predictive-rerun diagnostics."""

        return write_embedding_rerun_delta_manifest(self.config.workdir)


@dataclasses.dataclass(frozen=True)
class SiestaEwfResult:
    """Core-owned SIESTA block result prepared for an EWF collection step."""

    block_id: int
    machine_id: int | None
    rank: int | None
    core_atom_range: tuple[int, int]
    input_atom_range: tuple[int, int]
    core_atoms: tuple[int, ...]
    buffer_atoms: tuple[int, ...]
    core_atom_orbital_ranges: dict[int, tuple[int, int]]
    converged: bool | None
    total_energy_ev: float | None
    density_matrix_path: Path | None
    hamiltonian_matrix_path: Path | None
    overlap_matrix_path: Path | None
    orbital_index_path: Path | None
    output_path: Path | None
    matrix_metadata: dict[str, dict] = dataclasses.field(default_factory=dict)
    core_matrix_metadata: dict[str, dict] = dataclasses.field(default_factory=dict)
    run_diagnostics: dict[str, object] = dataclasses.field(default_factory=dict)

    def to_metadata(self) -> dict:
        return {
            "block_id": self.block_id,
            "machine_id": self.machine_id,
            "rank": self.rank,
            "core_atom_range": list(self.core_atom_range),
            "input_atom_range": list(self.input_atom_range),
            "core_atoms": list(self.core_atoms),
            "buffer_atoms": list(self.buffer_atoms),
            "core_atom_orbital_ranges": {
                str(atom): [start, end]
                for atom, (start, end) in sorted(self.core_atom_orbital_ranges.items())
            },
            "converged": self.converged,
            "total_energy_ev": self.total_energy_ev,
            "density_matrix_path": _path_to_str(self.density_matrix_path),
            "hamiltonian_matrix_path": _path_to_str(self.hamiltonian_matrix_path),
            "overlap_matrix_path": _path_to_str(self.overlap_matrix_path),
            "orbital_index_path": _path_to_str(self.orbital_index_path),
            "output_path": _path_to_str(self.output_path),
            "matrix_metadata": self.matrix_metadata,
            "core_matrix_metadata": self.core_matrix_metadata,
            "run_diagnostics": self.run_diagnostics,
        }

    def read_core_density_matrix(self) -> SiestaSparseCoreBlock:
        if self.density_matrix_path is None:
            raise ValueError("No density matrix path available")
        return read_density_matrix_sparse(self.density_matrix_path).core_block(self.core_atom_orbital_ranges)

    def read_core_hsx_matrix(self) -> SiestaHsxCoreBlock:
        if self.hamiltonian_matrix_path is None:
            raise ValueError("No HSX matrix path available")
        return read_hsx_sparse(self.hamiltonian_matrix_path).core_block(self.core_atom_orbital_ranges)


def parse_fdf(path: str | os.PathLike[str]) -> FdfInput:
    """Parse the coordinate block from a SIESTA FDF input."""

    lines = Path(path).read_text().splitlines()
    block_start = block_end = None
    in_coordinates = False
    in_species = False
    species_labels: dict[int, str] = {}
    species_atomic_numbers: dict[int, int] = {}
    atoms: list[Atom] = []

    for lineno, line in enumerate(lines):
        stripped = line.strip()
        lowered = stripped.lower()
        if lowered == "%block chemicalspecieslabel":
            in_species = True
            continue
        if lowered == "%endblock chemicalspecieslabel":
            in_species = False
            continue
        if in_species:
            fields = stripped.split()
            if len(fields) >= 3:
                try:
                    species_index = int(fields[0])
                    species_atomic_numbers[species_index] = int(fields[1])
                    species_labels[species_index] = fields[2]
                except ValueError:
                    pass
            continue
        if lowered == "%block atomiccoordinatesandatomicspecies":
            block_start = lineno
            in_coordinates = True
            continue
        if lowered == "%endblock atomiccoordinatesandatomicspecies":
            block_end = lineno
            in_coordinates = False
            continue
        if in_coordinates:
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 4:
                raise ValueError(f"Invalid coordinate line {lineno + 1}: {line!r}")
            atoms.append(
                Atom(
                    global_index=len(atoms),
                    x=float(fields[0]),
                    y=float(fields[1]),
                    z=float(fields[2]),
                    species=int(fields[3]),
                )
            )

    if block_start is None or block_end is None:
        raise ValueError("FDF input has no AtomicCoordinatesAndAtomicSpecies block")
    if not atoms:
        raise ValueError("FDF coordinate block is empty")
    return FdfInput(tuple(lines), tuple(atoms), block_start, block_end, species_labels, species_atomic_numbers)


def partition_contiguous_atoms(
    natoms: int,
    block_atoms: int,
    buffer_atoms: int = 0,
    num_machines: int | None = None,
) -> list[SiestaBlock]:
    """Split atoms into large contiguous blocks with optional atom-count buffer."""

    if natoms <= 0:
        raise ValueError("natoms must be positive")
    if block_atoms <= 0:
        raise ValueError("block_atoms must be positive")
    if buffer_atoms < 0:
        raise ValueError("buffer_atoms must be non-negative")

    blocks = []
    for block_id, core_start in enumerate(range(0, natoms, block_atoms)):
        core_end = min(core_start + block_atoms, natoms)
        machine_id = None if num_machines is None else block_id % num_machines
        blocks.append(
            SiestaBlock(
                block_id=block_id,
                core_atom_start=core_start,
                core_atom_end=core_end,
                input_atom_start=max(0, core_start - buffer_atoms),
                input_atom_end=min(natoms, core_end + buffer_atoms),
                machine_id=machine_id,
            )
        )
    return blocks


def partition_contiguous_atom_groups(
    natoms: int,
    group_size_atoms: int,
    block_groups: int,
    buffer_groups: int = 0,
    terminal_cap_atoms: int = 0,
    num_machines: int | None = None,
) -> list[SiestaBlock]:
    """Split an ordered chain into coarse blocks aligned to atom groups.

    This is useful for generated polyethylene inputs where one repeat group is
    ordered as `C C H H H H`.  If `terminal_cap_atoms=2`, one terminal atom is
    assigned to the first group and one to the final group.  The returned blocks
    still use contiguous atom ranges, but their boundaries are aligned to repeat
    groups rather than arbitrary atoms.
    """

    if natoms <= 0:
        raise ValueError("natoms must be positive")
    if group_size_atoms <= 0:
        raise ValueError("group_size_atoms must be positive")
    if block_groups <= 0:
        raise ValueError("block_groups must be positive")
    if buffer_groups < 0:
        raise ValueError("buffer_groups must be non-negative")
    if terminal_cap_atoms < 0:
        raise ValueError("terminal_cap_atoms must be non-negative")

    groups = _contiguous_atom_group_bounds(natoms, group_size_atoms, terminal_cap_atoms)
    ngroups = len(groups)

    blocks = []
    for block_id, core_group_start in enumerate(range(0, ngroups, block_groups)):
        core_group_end = min(core_group_start + block_groups, ngroups)
        input_group_start = max(0, core_group_start - buffer_groups)
        input_group_end = min(ngroups, core_group_end + buffer_groups)
        machine_id = None if num_machines is None else block_id % num_machines
        blocks.append(
            SiestaBlock(
                block_id=block_id,
                core_atom_start=groups[core_group_start][0],
                core_atom_end=groups[core_group_end - 1][1],
                input_atom_start=groups[input_group_start][0],
                input_atom_end=groups[input_group_end - 1][1],
                machine_id=machine_id,
            )
        )
    return blocks


def _contiguous_atom_group_bounds(
    natoms: int,
    group_size_atoms: int,
    terminal_cap_atoms: int = 0,
) -> list[tuple[int, int]]:
    if terminal_cap_atoms == 2 and natoms > group_size_atoms and (natoms - terminal_cap_atoms) % group_size_atoms == 0:
        repeat_groups = (natoms - terminal_cap_atoms) // group_size_atoms
        if repeat_groups >= 2:
            sizes = [group_size_atoms + 1]
            sizes.extend([group_size_atoms] * (repeat_groups - 2))
            sizes.append(group_size_atoms + 1)
            groups = []
            start = 0
            for size in sizes:
                end = start + size
                groups.append((start, end))
                start = end
            return groups

    groups = []
    full_groups, remainder = divmod(natoms, group_size_atoms)
    if full_groups == 0:
        groups.append((0, natoms))
    else:
        for group_id in range(full_groups):
            start = group_id * group_size_atoms
            end = start + group_size_atoms
            groups.append((start, end))
        if remainder:
            start, _ = groups[-1]
            groups[-1] = (start, natoms)
    return groups


def generate_block_directories(
    fdf: FdfInput,
    blocks: Sequence[SiestaBlock],
    workdir: str | os.PathLike[str],
    pseudopotentials: Iterable[str | os.PathLike[str]] = (),
    siesta_bin: str | None = None,
    threads_per_proc: int = 1,
    solver_config: SiestaSolverConfig | None = None,
) -> list[Path]:
    """Generate one SIESTA input directory per block."""

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    generated = []
    for block in blocks:
        block_dir = workdir / f"block_{block.block_id:04d}"
        block_dir.mkdir(parents=True, exist_ok=True)
        atoms = fdf.atoms[block.input_atom_start : block.input_atom_end]
        local_to_global = [atom.global_index for atom in atoms]

        solver_config = SiestaSolverConfig() if solver_config is None else solver_config
        (block_dir / "input.fdf").write_text(_render_block_fdf(fdf, atoms, block.block_id, solver_config))
        (block_dir / "block.json").write_text(
            json.dumps(block.to_metadata(local_to_global), indent=2, sort_keys=True) + "\n"
        )
        (block_dir / "solver_config.json").write_text(
            json.dumps(solver_config.to_metadata(), indent=2, sort_keys=True) + "\n"
        )
        (block_dir / "run.sh").write_text(_render_run_script(siesta_bin, threads_per_proc))
        (block_dir / "run.sh").chmod(0o755)

        for pseudo in pseudopotentials:
            pseudo = Path(pseudo)
            if pseudo.exists():
                shutil.copy2(pseudo, block_dir / pseudo.name)
        generated.append(block_dir)
    return generated


def prepare_siesta_workflow(
    fdf_path: str | os.PathLike[str],
    config: SiestaRunConfig | None = None,
    pseudopotentials: Iterable[str | os.PathLike[str]] = (),
) -> SiestaBlockWorkflow:
    """Prepare a reusable SIESTA block workflow from a global FDF input."""

    config = read_run_config() if config is None else config
    fdf = parse_fdf(fdf_path)
    blocks = build_blocks_from_config(fdf, config)
    block_dirs = tuple(config.workdir / f"block_{block.block_id:04d}" for block in blocks)
    return SiestaBlockWorkflow(
        fdf=fdf,
        config=config,
        blocks=tuple(blocks),
        block_dirs=block_dirs,
        pseudopotentials=tuple(Path(path) for path in pseudopotentials),
    )


def build_blocks_from_config(fdf: FdfInput, config: SiestaRunConfig) -> list[SiestaBlock]:
    """Build coarse SIESTA blocks from an FDF input and run configuration."""

    if config.group_size_atoms is not None or config.block_groups is not None:
        if config.group_size_atoms is None or config.block_groups is None:
            raise ValueError("EWF_GROUP_SIZE_ATOMS and EWF_BLOCK_GROUPS must be set together")
        return partition_contiguous_atom_groups(
            len(fdf.atoms),
            group_size_atoms=config.group_size_atoms,
            block_groups=config.block_groups,
            buffer_groups=config.buffer_groups,
            terminal_cap_atoms=config.terminal_cap_atoms,
            num_machines=config.num_machines,
        )
    block_atoms = config.block_atoms or default_block_atoms(
        len(fdf.atoms),
        config.num_machines,
        config.procs_per_machine,
    )
    return partition_contiguous_atoms(
        len(fdf.atoms),
        block_atoms=block_atoms,
        buffer_atoms=config.buffer_atoms,
        num_machines=config.num_machines,
    )


def infer_bonds(fdf: FdfInput, tolerance_angstrom: float = 0.45) -> list[dict]:
    """Infer covalent bonds from atom coordinates and species labels."""

    return _infer_bonds_grid(fdf, tolerance_angstrom=tolerance_angstrom)


def _infer_bonds_bruteforce(fdf: FdfInput, tolerance_angstrom: float = 0.45) -> list[dict]:
    """Infer covalent bonds by checking every atom pair."""

    bonds = []
    atoms = fdf.atoms
    for i, atom_i in enumerate(atoms):
        radius_i = _covalent_radius(fdf, atom_i.species)
        for atom_j in atoms[i + 1 :]:
            radius_j = _covalent_radius(fdf, atom_j.species)
            distance = _atom_distance(atom_i, atom_j)
            cutoff = radius_i + radius_j + tolerance_angstrom
            if distance <= cutoff:
                bonds.append(_bond_record(fdf, atom_i, atom_j, distance, cutoff))
    return bonds


def _infer_bonds_grid(fdf: FdfInput, tolerance_angstrom: float = 0.45) -> list[dict]:
    """Infer covalent bonds using fixed-size spatial bins."""

    atoms = fdf.atoms
    if not atoms:
        return []
    radii = [_covalent_radius(fdf, atom.species) for atom in atoms]
    cell_size = max(1.0e-12, 2.0 * max(radii) + tolerance_angstrom)
    bins: dict[tuple[int, int, int], list[int]] = {}
    for index, atom in enumerate(atoms):
        bins.setdefault(_atom_cell(atom, cell_size), []).append(index)

    bonds = []
    for i, atom_i in enumerate(atoms):
        radius_i = radii[i]
        cell = _atom_cell(atom_i, cell_size)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neighbor_cell = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                    for j in bins.get(neighbor_cell, []):
                        if j <= i:
                            continue
                        atom_j = atoms[j]
                        cutoff = radius_i + radii[j] + tolerance_angstrom
                        distance = _atom_distance(atom_i, atom_j)
                        if distance <= cutoff:
                            bonds.append(_bond_record(fdf, atom_i, atom_j, distance, cutoff))
    bonds.sort(key=lambda bond: (int(bond["atom_i"]), int(bond["atom_j"])))
    return bonds


def _atom_cell(atom: Atom, cell_size: float) -> tuple[int, int, int]:
    return (
        math.floor(atom.x / cell_size),
        math.floor(atom.y / cell_size),
        math.floor(atom.z / cell_size),
    )


def _bond_record(fdf: FdfInput, atom_i: Atom, atom_j: Atom, distance: float, cutoff: float) -> dict:
    return {
        "atom_i": atom_i.global_index,
        "atom_j": atom_j.global_index,
        "species_i": fdf.species_labels.get(atom_i.species, str(atom_i.species)),
        "species_j": fdf.species_labels.get(atom_j.species, str(atom_j.species)),
        "distance_angstrom": round(distance, 6),
        "cutoff_angstrom": round(cutoff, 6),
    }


def analyze_block_boundaries(fdf: FdfInput, blocks: Sequence[SiestaBlock]) -> dict:
    """Analyze bonds crossing block core boundaries and their buffer coverage."""

    bonds = infer_bonds(fdf)
    block_reports = []
    uncovered = []
    for block in blocks:
        core_atoms = set(range(block.core_atom_start, block.core_atom_end))
        input_atoms = set(range(block.input_atom_start, block.input_atom_end))
        boundary_bonds = []
        for bond in bonds:
            atom_i = int(bond["atom_i"])
            atom_j = int(bond["atom_j"])
            i_core = atom_i in core_atoms
            j_core = atom_j in core_atoms
            if i_core == j_core:
                continue
            outside_atom = atom_j if i_core else atom_i
            covered = outside_atom in input_atoms
            item = dict(bond)
            item["outside_core_atom"] = outside_atom
            item["covered_by_input"] = covered
            boundary_bonds.append(item)
            if not covered:
                uncovered.append({"block_id": block.block_id, **item})
        block_reports.append(
            {
                "block_id": block.block_id,
                "machine_id": block.machine_id,
                "core_atom_range": [block.core_atom_start, block.core_atom_end],
                "input_atom_range": [block.input_atom_start, block.input_atom_end],
                "num_boundary_bonds": len(boundary_bonds),
                "num_uncovered_boundary_bonds": sum(1 for bond in boundary_bonds if not bond["covered_by_input"]),
                "boundary_bonds": boundary_bonds,
            }
        )
    return {
        "num_atoms": len(fdf.atoms),
        "num_bonds": len(bonds),
        "num_blocks": len(blocks),
        "num_boundary_bonds": sum(block["num_boundary_bonds"] for block in block_reports),
        "num_uncovered_boundary_bonds": len(uncovered),
        "bonds": bonds,
        "blocks": block_reports,
        "uncovered_boundary_bonds": uncovered,
    }


def write_boundary_manifest(
    workdir: str | os.PathLike[str],
    fdf: FdfInput,
    blocks: Sequence[SiestaBlock],
) -> dict:
    """Write `boundary.json` with inferred bond-boundary diagnostics."""

    workdir = Path(workdir)
    payload = analyze_block_boundaries(fdf, blocks)
    (workdir / "boundary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def build_embedding_contract(boundary_payload: dict) -> dict:
    """Build the minimal boundary embedding contract from boundary diagnostics."""

    terms = []
    for block in boundary_payload.get("blocks", []):
        core_start, core_end = block["core_atom_range"]
        for bond in block.get("boundary_bonds", []):
            atom_i = int(bond["atom_i"])
            atom_j = int(bond["atom_j"])
            atom_i_core = core_start <= atom_i < core_end
            core_atom = atom_i if atom_i_core else atom_j
            environment_atom = atom_j if atom_i_core else atom_i
            covered = bool(bond.get("covered_by_input"))
            terms.append(
                {
                    "block_id": int(block["block_id"]),
                    "bond_atoms": [atom_i, atom_j],
                    "core_atom": core_atom,
                    "environment_atom": environment_atom,
                    "covered_by_input": covered,
                    "requires_embedding_potential": covered,
                    "requires_energy_correction": covered,
                    "status": "pending_embedding_correction" if covered else "invalid_uncovered_boundary",
                    "distance_angstrom": bond.get("distance_angstrom"),
                    "species": [bond.get("species_i"), bond.get("species_j")],
                }
            )
    uncovered = [term for term in terms if not term["covered_by_input"]]
    pending = [term for term in terms if term["status"] == "pending_embedding_correction"]
    return {
        "version": 1,
        "embedding_level": "boundary-buffer-contract",
        "matrix_ownership": "core_owned",
        "buffer_policy": "boundary_atoms_must_be_present_in_input",
        "electron_policy": "requires_global_electron_closure",
        "energy_policy": "requires_boundary_correction_closure",
        "num_terms": len(terms),
        "num_pending_embedding_terms": len(pending),
        "num_uncovered_boundary_terms": len(uncovered),
        "terms": terms,
    }


def write_embedding_contract_manifest(workdir: str | os.PathLike[str]) -> dict:
    """Write `embedding_contract.json` from an existing `boundary.json`."""

    workdir = Path(workdir)
    boundary_path = workdir / "boundary.json"
    if not boundary_path.exists():
        raise FileNotFoundError(boundary_path)
    payload = build_embedding_contract(json.loads(boundary_path.read_text()))
    (workdir / "embedding_contract.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def build_boundary_correction_plan(embedding_contract: dict, parameterize: bool = True) -> dict:
    """Build correction slots for each pending boundary embedding term.

    The default is a conservative minimal closure: the buffer atom is already
    included in the SIESTA block input, so the explicit boundary correction is a
    zero-valued bookkeeping term that marks the core-owned boundary as saturated
    by the local input rather than missing.  This is intentionally simple and
    machine-readable; higher-level EWF code can replace these values with a
    self-consistent embedding potential later.
    """

    corrections = []
    for term in embedding_contract.get("terms", []):
        if term.get("status") != "pending_embedding_correction":
            continue
        potential = None
        energy_correction = None
        electron_correction = None
        status = "not_parameterized"
        if parameterize:
            potential = {
                "model": "buffer_saturated_zero_shift",
                "value_ev": 0.0,
                "scope": "core_boundary_orbitals",
            }
            energy_correction = 0.0
            electron_correction = 0.0
            status = "parameterized"
        corrections.append(
            {
                "block_id": term["block_id"],
                "bond_atoms": term["bond_atoms"],
                "core_atom": term["core_atom"],
                "environment_atom": term["environment_atom"],
                "correction_type": "boundary_bond_embedding",
                "hamiltonian_embedding_potential": potential,
                "energy_correction_ev": energy_correction,
                "electron_count_correction": electron_correction,
                "status": status,
            }
        )
    unparameterized = sum(1 for correction in corrections if correction["status"] != "parameterized")
    return {
        "version": 1,
        "correction_level": "minimal-boundary-closure" if parameterize else "placeholder",
        "closure_model": "core-owned-buffer-saturated-zero-shift" if parameterize else None,
        "num_corrections": len(corrections),
        "num_parameterized_corrections": len(corrections) - unparameterized,
        "num_unparameterized_corrections": unparameterized,
        "corrections": corrections,
    }


def write_boundary_corrections_manifest(workdir: str | os.PathLike[str], parameterize: bool = True) -> dict:
    """Write `boundary_corrections.json` from `embedding_contract.json`."""

    workdir = Path(workdir)
    contract_path = workdir / "embedding_contract.json"
    if not contract_path.exists():
        raise FileNotFoundError(contract_path)
    payload = build_boundary_correction_plan(json.loads(contract_path.read_text()), parameterize=parameterize)
    (workdir / "boundary_corrections.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def calibrate_boundary_corrections_to_reference(
    workdir: str | os.PathLike[str],
    reference_total_energy_ev: float,
) -> dict:
    """Calibrate boundary energy corrections so embedded energy matches a reference."""

    workdir = Path(workdir)
    corrections_path = workdir / "boundary_corrections.json"
    observables_path = workdir / "embedded_observables.json"
    if not corrections_path.exists():
        raise FileNotFoundError(corrections_path)
    if not observables_path.exists():
        raise FileNotFoundError(observables_path)
    corrections = json.loads(corrections_path.read_text())
    observables = json.loads(observables_path.read_text())
    validation = _read_optional_json(workdir / "validation.json") or {}
    current_energy = validation.get("total_block_energy_ev", observables.get("total_block_energy_ev"))
    baseline_source = "validation.total_block_energy_ev" if current_energy is not None else None
    if current_energy is None:
        current_energy = observables.get("embedded_total_energy_ev")
        baseline_source = "embedded_observables.embedded_total_energy_ev"
    if current_energy is None:
        raise ValueError("No total block or embedded energy is available for calibration")
    correction_slots = corrections.get("corrections", [])
    if not correction_slots:
        raise ValueError("No boundary correction slots are available for calibration")
    delta = float(reference_total_energy_ev) - float(current_energy)
    per_slot = delta / len(correction_slots)
    for correction in correction_slots:
        correction["energy_correction_ev"] = float(per_slot)
        correction["calibration"] = {
            "model": "reference_total_energy_match",
            "reference_total_energy_ev": float(reference_total_energy_ev),
            "baseline_total_energy_ev": float(current_energy),
            "baseline_source": baseline_source,
        }
        correction["status"] = "parameterized"
    corrections["correction_level"] = "reference-calibrated-boundary-closure"
    corrections["closure_model"] = "reference-total-energy-matched-boundary-shift"
    corrections["num_parameterized_corrections"] = len(correction_slots)
    corrections["num_unparameterized_corrections"] = 0
    corrections["reference_total_energy_ev"] = float(reference_total_energy_ev)
    corrections["calibration_baseline_total_energy_ev"] = float(current_energy)
    corrections["calibration_baseline_source"] = baseline_source
    corrections["total_calibrated_energy_correction_ev"] = float(delta)
    corrections_path.write_text(json.dumps(corrections, indent=2, sort_keys=True) + "\n")
    return corrections


def build_predictive_boundary_potential(
    workdir: str | os.PathLike[str],
    damping: float = 1.0,
) -> dict:
    """Build a non-reference boundary potential from SIESTA DM/HSX couplings."""

    workdir = Path(workdir)
    contract = _read_optional_json(workdir / "embedding_contract.json") or {}
    results = _read_json_list(workdir / "results.json") if (workdir / "results.json").exists() else collect_rank_results(workdir)
    results_by_block = {int(result["block_id"]): result for result in results}
    terms = []
    total_energy_correction = 0.0
    blockers = []
    for term in contract.get("terms", []):
        if term.get("status") != "pending_embedding_correction":
            continue
        block_id = int(term["block_id"])
        result = results_by_block.get(block_id)
        if result is None:
            blockers.append(f"Block {block_id} has no SIESTA result for predictive boundary potential")
            terms.append(_predictive_boundary_error_term(term, "missing_result"))
            continue
        try:
            coupling = _compute_boundary_coupling_from_result(
                result,
                int(term["core_atom"]),
                int(term["environment_atom"]),
            )
        except Exception as exc:
            blockers.append(f"Block {block_id} boundary {term['bond_atoms']} could not be evaluated: {exc}")
            terms.append(_predictive_boundary_error_term(term, str(exc)))
            continue
        energy_correction = float(-0.5 * damping * coupling["density_hamiltonian_coupling_ev"])
        potential_value = float(
            -damping
            * coupling["density_hamiltonian_coupling_ev"]
            / max(1, int(coupling["num_core_orbitals"]))
        )
        sparse_entries = []
        for entry in coupling["sparse_hamiltonian_embedding_entries"]:
            scaled = dict(entry)
            scaled["value_ev"] = float(damping * scaled["value_ev"])
            sparse_entries.append(scaled)
        total_energy_correction += energy_correction
        terms.append(
            {
                "block_id": block_id,
                "bond_atoms": list(term["bond_atoms"]),
                "core_atom": int(term["core_atom"]),
                "environment_atom": int(term["environment_atom"]),
                "status": "parameterized",
                "model": "density_hamiltonian_boundary_coupling_v1",
                "source": "siesta_returned_dm_hsx",
                "damping": float(damping),
                "num_core_orbitals": coupling["num_core_orbitals"],
                "num_environment_orbitals": coupling["num_environment_orbitals"],
                "num_coupling_entries": coupling["num_coupling_entries"],
                "density_hamiltonian_coupling_ev": coupling["density_hamiltonian_coupling_ev"],
                "density_overlap_population": coupling["density_overlap_population"],
                "sparse_hamiltonian_embedding_potential": {
                    "model": "boundary-hsx-nonlocal-shift-v1",
                    "scope": "core_environment_sparse_couplings",
                    "entries": sparse_entries,
                    "num_entries": len(sparse_entries),
                    "applied_to_siesta_input": False,
                },
                "hamiltonian_embedding_potential": {
                    "model": "average_core_boundary_shift_from_density_hamiltonian_coupling",
                    "value_ev": potential_value,
                    "scope": "core_boundary_orbitals",
                    "applied_to_siesta_input": False,
                },
                "energy_correction_ev": energy_correction,
                "electron_count_correction": 0.0,
            }
        )
    unparameterized = sum(1 for term in terms if term.get("status") != "parameterized")
    siesta_applied = _siesta_embedding_potential_applied(workdir)
    all_converged = bool(results) and all(result.get("converged") is True for result in results)
    if terms and not blockers and not siesta_applied:
        blockers.append(
            "Predictive boundary potential is computed from SIESTA-returned DM/HSX but is not yet injected back into SIESTA Hamiltonian for a self-consistent rerun"
        )
    if siesta_applied and not all_converged:
        blockers.append("SIESTA embedding potential was applied, but not all block reruns converged")
    self_consistency_status = (
        "converged"
        if siesta_applied and all_converged
        else "external_potential_applied_not_converged"
        if siesta_applied
        else "single_shot_not_self_consistent"
    )
    return {
        "version": 1,
        "potential_level": "predictive-boundary-potential-v1",
        "model": "density_hamiltonian_boundary_coupling_v1",
        "source": "siesta_returned_dm_hsx",
        "uses_reference_energy": False,
        "damping": float(damping),
        "self_consistency_status": self_consistency_status,
        "sIESTA_external_potential_applied": siesta_applied,
        "num_terms": len(terms),
        "num_parameterized_terms": len(terms) - unparameterized,
        "num_unparameterized_terms": unparameterized,
        "total_predictive_energy_correction_ev": float(total_energy_correction),
        "blockers": blockers,
        "terms": terms,
    }


def write_predictive_boundary_potential_manifest(
    workdir: str | os.PathLike[str],
    damping: float = 1.0,
) -> dict:
    """Write `predictive_embedding_potential.json` from returned SIESTA matrices."""

    workdir = Path(workdir)
    payload = build_predictive_boundary_potential(workdir, damping=damping)
    (workdir / "predictive_embedding_potential.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload


def write_predictive_boundary_corrections_manifest(
    workdir: str | os.PathLike[str],
    damping: float = 1.0,
) -> dict:
    """Write predictive non-reference boundary corrections from SIESTA DM/HSX."""

    workdir = Path(workdir)
    potential = write_predictive_boundary_potential_manifest(workdir, damping=damping)
    corrections = []
    for term in potential.get("terms", []):
        if term.get("status") != "parameterized":
            continue
        corrections.append(
            {
                "block_id": term["block_id"],
                "bond_atoms": term["bond_atoms"],
                "core_atom": term["core_atom"],
                "environment_atom": term["environment_atom"],
                "correction_type": "predictive_boundary_bond_embedding",
                "hamiltonian_embedding_potential": term["hamiltonian_embedding_potential"],
                "energy_correction_ev": term["energy_correction_ev"],
                "electron_count_correction": term["electron_count_correction"],
                "status": "parameterized",
                "model": term["model"],
                "source": term["source"],
                "density_hamiltonian_coupling_ev": term["density_hamiltonian_coupling_ev"],
                "density_overlap_population": term["density_overlap_population"],
                "sparse_hamiltonian_embedding_potential": term.get("sparse_hamiltonian_embedding_potential"),
            }
        )
    payload = {
        "version": 1,
        "correction_level": "predictive-boundary-coupling-v1",
        "closure_model": "density-hamiltonian-boundary-coupling-v1",
        "uses_reference_energy": False,
        "self_consistency_status": potential.get("self_consistency_status"),
        "sIESTA_external_potential_applied": potential.get("sIESTA_external_potential_applied"),
        "num_corrections": len(corrections),
        "num_parameterized_corrections": len(corrections),
        "num_unparameterized_corrections": int(potential.get("num_unparameterized_terms", 0)),
        "total_predictive_energy_correction_ev": potential.get("total_predictive_energy_correction_ev"),
        "blockers": list(potential.get("blockers", [])),
        "corrections": corrections,
    }
    (workdir / "boundary_corrections.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def build_predictive_ewf_closure(
    workdir: str | os.PathLike[str],
    bath_threshold: float = 1.0e-6,
) -> dict:
    """Build an unreferenced EWF closure diagnostic from returned SIESTA matrices.

    This is deliberately a mean-field closure layer: it constructs boundary bath
    diagnostics and an external-potential double-counting term from the returned
    block DM/HSX data, but it does not claim that a correlated solver has been
    run.
    """

    workdir = Path(workdir)
    validation = _read_optional_json(workdir / "validation.json") or {}
    corrections = _read_optional_json(workdir / "boundary_corrections.json") or {}
    predictive = _read_optional_json(workdir / "predictive_embedding_potential.json") or {}
    observables = _read_optional_json(workdir / "embedded_observables.json") or {}
    results = _read_json_list(workdir / "results.json") if (workdir / "results.json").exists() else collect_rank_results(workdir)
    results_by_block = {int(result["block_id"]): result for result in results}

    bath_terms = []
    blockers = []
    predictive_terms = [
        term
        for term in predictive.get("terms", [])
        if term.get("status") == "parameterized"
    ]
    closure_terms = predictive_terms or corrections.get("corrections", [])
    for term in closure_terms:
        block_id = int(term.get("block_id", -1))
        result = results_by_block.get(block_id)
        if result is None:
            blockers.append(f"Block {block_id} has no SIESTA result for bath construction")
            continue
        try:
            bath_terms.append(
                _compute_boundary_bath_from_result(
                    result,
                    int(term["core_atom"]),
                    int(term["environment_atom"]),
                    threshold=bath_threshold,
                )
            )
        except Exception as exc:
            blockers.append(f"Block {block_id} bath term {term.get('bond_atoms')} could not be evaluated: {exc}")

    potential_terms = []
    for result in results:
        block_id = int(result["block_id"])
        block_dir = workdir / f"block_{block_id:04d}"
        potential_path = block_dir / "ewf_embedding_potential.dat"
        if not potential_path.exists():
            potential_terms.append(
                {
                    "block_id": block_id,
                    "potential_file": str(potential_path),
                    "present": False,
                    "embedding_potential_expectation_ev": None,
                    "matched_entries": 0,
                    "missing_density_entries": 0,
                }
            )
            continue
        potential_terms.append(_compute_embedding_potential_expectation(result, potential_path))

    potential_expectations = [
        float(term["embedding_potential_expectation_ev"])
        for term in potential_terms
        if term.get("embedding_potential_expectation_ev") is not None
    ]
    total_potential_expectation = float(sum(potential_expectations))
    boundary_energy = (
        predictive.get("total_predictive_energy_correction_ev")
        if predictive_terms and predictive.get("uses_reference_energy") is False
        else observables.get("boundary_energy_correction_ev")
    )
    total_block_energy = validation.get("total_block_energy_ev", observables.get("total_block_energy_ev"))
    predictive_total_energy = None
    double_counting_correction = None
    if total_block_energy is not None and boundary_energy is not None:
        double_counting_correction = float(-total_potential_expectation)
        predictive_total_energy = float(total_block_energy) + float(boundary_energy) + double_counting_correction

    all_bath_ready = bool(bath_terms) and not blockers and all(term.get("bath_rank", 0) > 0 for term in bath_terms)
    all_potential_expectations_ready = bool(potential_terms) and all(
        term.get("embedding_potential_expectation_ev") is not None for term in potential_terms
    )
    predictive_ready = bool(
        validation.get("ok")
        and predictive.get("self_consistency_status") == "converged"
        and predictive.get("sIESTA_external_potential_applied") is True
        and all_bath_ready
        and all_potential_expectations_ready
    )
    if not all_bath_ready:
        blockers.append("At least one boundary bath term is missing or rank deficient")
    if not all_potential_expectations_ready:
        blockers.append("Embedding-potential expectation values are incomplete")

    return {
        "version": 1,
        "closure_level": "predictive-mean-field-ewf-closure-v1",
        "status": "ready" if predictive_ready else "diagnostic_incomplete",
        "uses_reference_energy": False,
        "production_predictive_physics_ready": False,
        "production_blockers": [
            "No correlated fragment solver has been run on the constructed bath space",
            "No chemical-potential iteration across fragments is implemented",
            "Double-counting correction is a mean-field external-potential expectation diagnostic",
        ],
        "correlated_solver_status": "not_run_mean_field_surrogate_only",
        "bath_construction": {
            "model": "boundary-density-svd-v1",
            "threshold": float(bath_threshold),
            "num_terms": len(bath_terms),
            "total_bath_rank": int(sum(int(term.get("bath_rank", 0)) for term in bath_terms)),
            "terms": bath_terms,
        },
        "double_counting": {
            "model": "subtract-external-embedding-potential-expectation-v1",
            "embedding_potential_expectation_ev": total_potential_expectation,
            "energy_correction_ev": double_counting_correction,
            "terms": potential_terms,
        },
        "energy": {
            "policy": "block_sum_plus_boundary_correction_minus_embedding_potential_expectation",
            "total_block_energy_ev": total_block_energy,
            "boundary_energy_correction_ev": boundary_energy,
            "double_counting_energy_correction_ev": double_counting_correction,
            "predictive_total_energy_ev": predictive_total_energy,
        },
        "blockers": blockers,
    }


def write_predictive_ewf_closure_manifest(
    workdir: str | os.PathLike[str],
    bath_threshold: float = 1.0e-6,
) -> dict:
    """Write `predictive_ewf_closure.json` for the unreferenced EWF closure layer."""

    workdir = Path(workdir)
    payload = build_predictive_ewf_closure(workdir, bath_threshold=bath_threshold)
    (workdir / "predictive_ewf_closure.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def build_cluster_hamiltonians(
    workdir: str | os.PathLike[str],
    bath_threshold: float = 1.0e-6,
    overlap_eigenvalue_floor: float = 1.0e-8,
) -> dict:
    """Build solver-ready block cluster Hamiltonian artifacts.

    The current artifact is a one-electron, SIESTA-AO based cluster input.  It
    writes a compact NPZ per block containing the projected cluster basis,
    Lowdin orthogonalizer, and transformed H/S/DM arrays.  A later correlated
    solver can consume these files without reparsing SIESTA Fortran binaries.
    """

    workdir = Path(workdir)
    blocks = _read_json_list(workdir / "blocks.json")
    results = _read_json_list(workdir / "results.json") if (workdir / "results.json").exists() else collect_rank_results(workdir)
    results_by_block = {int(result["block_id"]): result for result in results}
    closure = _read_optional_json(workdir / "predictive_ewf_closure.json") or {}
    block_payloads = []
    blockers = []
    for block in blocks:
        block_id = int(block["block_id"])
        result = results_by_block.get(block_id)
        if result is None:
            blockers.append(f"Block {block_id} has no SIESTA result for cluster Hamiltonian")
            continue
        try:
            block_payloads.append(
                _write_cluster_hamiltonian_for_block(
                    workdir,
                    block,
                    result,
                    closure,
                    bath_threshold=bath_threshold,
                    overlap_eigenvalue_floor=overlap_eigenvalue_floor,
                )
            )
        except Exception as exc:
            blockers.append(f"Block {block_id} cluster Hamiltonian failed: {exc}")
    payload = {
        "version": 1,
        "artifact_level": "solver-ready-one-electron-cluster-v1",
        "basis_model": "core-ao-plus-boundary-density-svd-bath",
        "orthogonalization": "lowdin",
        "uses_reference_energy": False,
        "num_blocks": len(blocks),
        "num_written_blocks": len(block_payloads),
        "ready": len(block_payloads) == len(blocks) and not blockers,
        "bath_threshold": float(bath_threshold),
        "overlap_eigenvalue_floor": float(overlap_eigenvalue_floor),
        "blockers": blockers,
        "blocks": block_payloads,
    }
    (workdir / "cluster_hamiltonians.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def write_cluster_hamiltonians_manifest(
    workdir: str | os.PathLike[str],
    bath_threshold: float = 1.0e-6,
    overlap_eigenvalue_floor: float = 1.0e-8,
) -> dict:
    """Write `cluster_hamiltonians.json` and per-block cluster NPZ files."""

    return build_cluster_hamiltonians(
        workdir,
        bath_threshold=bath_threshold,
        overlap_eigenvalue_floor=overlap_eigenvalue_floor,
    )


def solve_cluster_hamiltonians(
    workdir: str | os.PathLike[str],
) -> dict:
    """Consume cluster Hamiltonian NPZ files with a one-electron reference solver."""

    workdir = Path(workdir)
    clusters = _read_optional_json(workdir / "cluster_hamiltonians.json") or {}
    blocks = []
    blockers = []
    for block in clusters.get("blocks", []):
        try:
            blocks.append(_solve_one_cluster_hamiltonian(block))
        except Exception as exc:
            block_id = block.get("block_id", "unknown")
            blockers.append(f"Block {block_id} cluster solver failed: {exc}")
    total_density_energy = [
        block["density_projected_one_electron_energy_ev"]
        for block in blocks
        if block.get("density_projected_one_electron_energy_ev") is not None
    ]
    total_aufbau_energy = [
        block["aufbau_one_electron_energy_ev"]
        for block in blocks
        if block.get("aufbau_one_electron_energy_ev") is not None
    ]
    payload = {
        "version": 1,
        "solver_level": "one-electron-lowdin-cluster-reference-v1",
        "solver_kind": "one_electron_reference",
        "uses_reference_energy": False,
        "correlated_solver_status": "not_run_one_electron_reference_only",
        "production_predictive_physics_ready": False,
        "num_blocks": int(clusters.get("num_blocks", len(blocks))),
        "num_solved_blocks": len(blocks),
        "ready": len(blocks) == int(clusters.get("num_blocks", len(blocks))) and not blockers,
        "total_density_projected_one_electron_energy_ev": float(sum(total_density_energy))
        if total_density_energy
        else None,
        "total_aufbau_one_electron_energy_ev": float(sum(total_aufbau_energy)) if total_aufbau_energy else None,
        "blockers": blockers,
        "blocks": blocks,
    }
    (workdir / "cluster_solver_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def write_cluster_solver_results_manifest(workdir: str | os.PathLike[str]) -> dict:
    """Write `cluster_solver_results.json` by solving cluster Hamiltonian NPZ files."""

    return solve_cluster_hamiltonians(workdir)


def solve_effective_interaction_clusters(
    workdir: str | os.PathLike[str],
    effective_interaction_u_ev: float = 0.0,
    denominator_shift_ev: float = 1.0e-6,
) -> dict:
    """Run a model correlated solver on solved cluster Hamiltonians.

    This is an explicit prototype for plumbing correlated corrections through
    the EWF/SIESTA path.  It uses a local Hubbard-like effective interaction in
    the Lowdin cluster basis; it is not an ab-initio two-electron integral path.
    """

    workdir = Path(workdir)
    clusters = _read_optional_json(workdir / "cluster_hamiltonians.json") or {}
    cluster_solver = _read_optional_json(workdir / "cluster_solver_results.json") or {}
    blocks = []
    blockers = []
    for block in clusters.get("blocks", []):
        try:
            blocks.append(
                _solve_effective_interaction_cluster_block(
                    block,
                    effective_interaction_u_ev=float(effective_interaction_u_ev),
                    denominator_shift_ev=float(denominator_shift_ev),
                )
            )
        except Exception as exc:
            block_id = block.get("block_id", "unknown")
            blockers.append(f"Block {block_id} effective interaction solver failed: {exc}")
    corrections = [
        block["correlation_energy_ev"]
        for block in blocks
        if block.get("correlation_energy_ev") is not None
    ]
    payload = {
        "version": 1,
        "solver_level": "effective-interaction-second-order-cluster-v1",
        "solver_kind": "model_correlated_effective_interaction",
        "interaction_model": "local-hubbard-like-lowdin-orbital-overlap",
        "uses_ab_initio_two_electron_integrals": False,
        "uses_reference_energy": False,
        "effective_interaction_u_ev": float(effective_interaction_u_ev),
        "denominator_shift_ev": float(denominator_shift_ev),
        "input_cluster_solver_level": cluster_solver.get("solver_level"),
        "num_blocks": int(clusters.get("num_blocks", len(blocks))),
        "num_solved_blocks": len(blocks),
        "ready": len(blocks) == int(clusters.get("num_blocks", len(blocks))) and not blockers,
        "correlated_solver_status": "model_effective_interaction_solved",
        "production_predictive_physics_ready": False,
        "production_blockers": [
            "Effective interaction U is model supplied, not derived from SIESTA two-electron integrals",
            "Correlation formula is a second-order prototype, not a validated EWF solver kernel",
            "No chemical-potential self-consistency loop is coupled to this correction",
        ],
        "total_correlation_energy_ev": float(sum(corrections)) if corrections else None,
        "blockers": blockers,
        "blocks": blocks,
    }
    (workdir / "effective_correlated_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def write_effective_correlated_results_manifest(
    workdir: str | os.PathLike[str],
    effective_interaction_u_ev: float = 0.0,
    denominator_shift_ev: float = 1.0e-6,
) -> dict:
    """Write `effective_correlated_results.json` from cluster Hamiltonian NPZ files."""

    return solve_effective_interaction_clusters(
        workdir,
        effective_interaction_u_ev=effective_interaction_u_ev,
        denominator_shift_ev=denominator_shift_ev,
    )


def build_effective_interaction_benchmark_scan(
    workdir: str | os.PathLike[str],
    reference_observables: dict,
    u_values_ev: Sequence[float],
    denominator_shift_ev: float = 1.0e-6,
) -> dict:
    """Scan model effective-interaction U values against a reference observable."""

    workdir = Path(workdir)
    if not u_values_ev:
        raise ValueError("At least one U value is required")
    embedded = _read_optional_json(workdir / "embedded_observables.json") or {}
    baseline_energy = embedded.get("embedded_total_energy_ev")
    reference_energy = reference_observables.get("total_energy_ev")
    if baseline_energy is None:
        raise ValueError("embedded_observables.json has no embedded_total_energy_ev")
    if reference_energy is None:
        raise ValueError("reference observables have no total_energy_ev")
    blocks = (_read_optional_json(workdir / "cluster_hamiltonians.json") or {}).get("blocks", [])
    samples = []
    for u_value in u_values_ev:
        block_results = [
            _solve_effective_interaction_cluster_block(
                block,
                effective_interaction_u_ev=float(u_value),
                denominator_shift_ev=float(denominator_shift_ev),
            )
            for block in blocks
        ]
        correlation = float(sum(block["correlation_energy_ev"] for block in block_results))
        total = float(baseline_energy) + correlation
        error = total - float(reference_energy)
        samples.append(
            {
                "effective_interaction_u_ev": float(u_value),
                "total_correlation_energy_ev": correlation,
                "effective_embedded_total_energy_ev": total,
                "energy_error_ev": error,
                "abs_energy_error_ev": abs(error),
            }
        )
    best = min(samples, key=lambda item: item["abs_energy_error_ev"])
    unit_correlation = None
    try:
        unit_blocks = [
            _solve_effective_interaction_cluster_block(
                block,
                effective_interaction_u_ev=1.0,
                denominator_shift_ev=float(denominator_shift_ev),
            )
            for block in blocks
        ]
        unit_correlation = float(sum(block["correlation_energy_ev"] for block in unit_blocks))
    except Exception:
        unit_correlation = None
    required_correlation = float(reference_energy) - float(baseline_energy)
    fit_u = None
    fit_possible = False
    fit_reason = None
    if unit_correlation in (None, 0.0):
        fit_reason = "unit-U correlation is zero or unavailable"
    else:
        ratio = required_correlation / unit_correlation
        if ratio >= 0.0:
            fit_u = float(math.sqrt(ratio))
            fit_possible = True
            fit_reason = "real nonnegative U can match the reference energy in this quadratic model"
        else:
            fit_reason = "required correction has opposite sign from the U^2 effective-interaction model"
    return {
        "version": 1,
        "benchmark_level": "effective_interaction_u_scan_vs_reference",
        "reference_label": reference_observables.get("label", "reference"),
        "reference_kind": reference_observables.get("reference_kind", "external_reference"),
        "reference_is_external": bool(reference_observables.get("reference_is_external", True)),
        "uses_ab_initio_two_electron_integrals": False,
        "baseline_embedded_total_energy_ev": float(baseline_energy),
        "reference_total_energy_ev": float(reference_energy),
        "baseline_energy_error_ev": float(baseline_energy) - float(reference_energy),
        "denominator_shift_ev": float(denominator_shift_ev),
        "num_samples": len(samples),
        "samples": samples,
        "best_sample": best,
        "unit_u_total_correlation_energy_ev": unit_correlation,
        "reference_required_correlation_energy_ev": required_correlation,
        "reference_fit_possible_with_real_nonnegative_u": fit_possible,
        "reference_fit_u_ev": fit_u,
        "reference_fit_reason": fit_reason,
        "status": "scan_complete",
    }


def write_effective_interaction_benchmark_scan_manifest(
    workdir: str | os.PathLike[str],
    reference_observables: dict,
    u_values_ev: Sequence[float],
    denominator_shift_ev: float = 1.0e-6,
) -> dict:
    """Write `effective_interaction_benchmark_scan.json`."""

    workdir = Path(workdir)
    payload = build_effective_interaction_benchmark_scan(
        workdir,
        reference_observables,
        u_values_ev=u_values_ev,
        denominator_shift_ev=denominator_shift_ev,
    )
    (workdir / "effective_interaction_benchmark_scan.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload


def write_siesta_embedding_potential_inputs(workdir: str | os.PathLike[str]) -> dict:
    """Write block-local SIESTA embedding potential files and FDF hooks."""

    workdir = Path(workdir)
    potential = _read_optional_json(workdir / "predictive_embedding_potential.json")
    if potential is None:
        potential = write_predictive_boundary_potential_manifest(workdir)
    results = _read_json_list(workdir / "results.json") if (workdir / "results.json").exists() else collect_rank_results(workdir)
    results_by_block = {int(result["block_id"]): result for result in results}
    blocks = _read_json_list(workdir / "blocks.json")
    written_blocks = []
    skipped_terms = []
    for block in blocks:
        block_id = int(block["block_id"])
        block_dir = workdir / f"block_{block_id:04d}"
        result = results_by_block.get(block_id)
        if result is None:
            continue
        atom_ranges = {
            int(atom): (int(bounds[0]), int(bounds[1]))
            for atom, bounds in result.get("atom_orbital_ranges", {}).items()
        }
        entries: dict[tuple[int, int, int], float] = {}
        diagonal_fallback_terms = 0
        sparse_terms = 0
        for term in potential.get("terms", []):
            if int(term.get("block_id", -1)) != block_id or term.get("status") != "parameterized":
                continue
            core_atom = int(term["core_atom"])
            if core_atom not in atom_ranges:
                skipped_terms.append({"block_id": block_id, "core_atom": core_atom, "reason": "missing_orbital_range"})
                continue
            sparse_potential = term.get("sparse_hamiltonian_embedding_potential") or {}
            sparse_entries = sparse_potential.get("entries") or []
            if sparse_entries:
                sparse_terms += 1
                for entry in sparse_entries:
                    key = (int(entry["row"]) + 1, int(entry["col"]) + 1, int(entry["spin"]) + 1)
                    entries[key] = entries.get(key, 0.0) + float(entry["value_ev"])
                continue
            diagonal_fallback_terms += 1
            value = float(term["hamiltonian_embedding_potential"]["value_ev"])
            start, end = atom_ranges[core_atom]
            nspin = max(1, _nspin_hamiltonian_from_result(result))
            for orbital in range(start + 1, end + 1):
                for spin in range(1, nspin + 1):
                    key = (orbital, orbital, spin)
                    entries[key] = entries.get(key, 0.0) + value
        if not entries:
            continue
        potential_path = block_dir / "ewf_embedding_potential.dat"
        lines = [
            "# row col spin value_eV",
            "# generated from predictive_embedding_potential.json",
        ]
        for (row, col, spin), value in sorted(entries.items()):
            lines.append(f"{row:d} {col:d} {spin:d} {value:.16e}")
        potential_path.write_text("\n".join(lines) + "\n")
        _ensure_fdf_option(block_dir / "input.fdf", "EWF.Embedding.PotentialFile", potential_path.name)
        written_blocks.append(
            {
                "block_id": block_id,
                "potential_file": str(potential_path),
                "num_entries": len(entries),
                "sum_value_ev": float(sum(entries.values())),
                "num_sparse_terms": sparse_terms,
                "num_diagonal_fallback_terms": diagonal_fallback_terms,
            }
        )
    payload = {
        "version": 1,
        "model": "sparse-nonlocal-boundary-shift-v1",
        "fallback_model": "diagonal-core-orbital-boundary-shift-v1",
        "source": "predictive_embedding_potential.json",
        "sIESTA_fdf_key": "EWF.Embedding.PotentialFile",
        "num_blocks_with_potential": len(written_blocks),
        "blocks": written_blocks,
        "skipped_terms": skipped_terms,
    }
    (workdir / "siesta_embedding_potential_inputs.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload


def assign_blocks_to_machine(blocks: Sequence[SiestaBlock], machine_id: int) -> list[SiestaBlock]:
    """Return blocks assigned to a simulated machine."""

    return [block for block in blocks if block.machine_id == machine_id]


def assign_blocks_to_rank(
    blocks: Sequence[SiestaBlock],
    machine_id: int,
    local_rank: int,
    procs_per_machine: int,
) -> list[SiestaBlock]:
    """Return blocks assigned to one rank within a simulated machine."""

    if procs_per_machine <= 0:
        raise ValueError("procs_per_machine must be positive")
    if local_rank < 0 or local_rank >= procs_per_machine:
        raise ValueError("local_rank must be in [0, procs_per_machine)")

    machine_blocks = assign_blocks_to_machine(blocks, machine_id)
    return [
        block
        for offset, block in enumerate(machine_blocks)
        if offset % procs_per_machine == local_rank
    ]


def build_schedule(
    blocks: Sequence[SiestaBlock],
    num_machines: int,
    procs_per_machine: int,
    threads_per_proc: int | None = None,
) -> dict:
    """Build the complete block-to-rank schedule for a simulated topology."""

    if num_machines <= 0:
        raise ValueError("num_machines must be positive")
    if procs_per_machine <= 0:
        raise ValueError("procs_per_machine must be positive")
    ranks = []
    for machine_id in range(num_machines):
        for local_rank in range(procs_per_machine):
            rank = machine_id * procs_per_machine + local_rank
            assigned = assign_blocks_to_rank(blocks, machine_id, local_rank, procs_per_machine)
            ranks.append(
                {
                    "rank": rank,
                    "machine_id": machine_id,
                    "local_rank": local_rank,
                    "num_blocks": len(assigned),
                    "block_ids": [block.block_id for block in assigned],
                    "blocks": [_schedule_block_metadata(block) for block in assigned],
                }
            )
    block_owner = {}
    for rank_info in ranks:
        for block_id in rank_info["block_ids"]:
            block_owner[str(block_id)] = rank_info["rank"]
    rank_block_counts = {str(rank_info["rank"]): rank_info["num_blocks"] for rank_info in ranks}
    total_ranks = num_machines * procs_per_machine
    return {
        "num_machines": num_machines,
        "procs_per_machine": procs_per_machine,
        "threads_per_proc": threads_per_proc,
        "num_ranks": total_ranks,
        "total_ranks": total_ranks,
        "num_blocks": len(blocks),
        "rank_block_counts": rank_block_counts,
        "ranks": ranks,
        "block_owner_rank": block_owner,
    }


def write_schedule_manifest(
    workdir: str | os.PathLike[str],
    blocks: Sequence[SiestaBlock],
    num_machines: int,
    procs_per_machine: int,
    threads_per_proc: int | None = None,
) -> dict:
    """Write `schedule.json` with the complete planned block distribution."""

    workdir = Path(workdir)
    payload = build_schedule(blocks, num_machines, procs_per_machine, threads_per_proc=threads_per_proc)
    (workdir / "schedule.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def write_block_manifest(workdir: str | os.PathLike[str], block_dirs: Sequence[Path]) -> list[dict]:
    """Write `blocks.json` from generated block directory metadata."""

    workdir = Path(workdir)
    blocks = []
    for block_dir in block_dirs:
        metadata_path = block_dir / "block.json"
        if metadata_path.exists():
            blocks.append(json.loads(metadata_path.read_text()))
    (workdir / "blocks.json").write_text(json.dumps(blocks, indent=2, sort_keys=True) + "\n")
    return blocks


def write_rank_results(
    workdir: str | os.PathLike[str],
    rank: int,
    parsed_results: Sequence[SiestaResult],
    run_results: Sequence[subprocess.CompletedProcess],
) -> list[dict]:
    """Write `result_rank_XXXX.json` from one rank's completed block runs."""

    workdir = Path(workdir)
    payload = []
    for parsed, completed in zip(parsed_results, run_results):
        result = parsed.to_metadata()
        result["returncode"] = completed.returncode
        payload.append(result)
    (workdir / f"result_rank_{rank:04d}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def add_siesta_block_fragments(
    fragmentation,
    blocks: Sequence[SiestaBlock],
    orbital_filter=None,
    name_prefix: str = "siesta-block",
):
    """Add SIESTA coarse blocks as Vayesta atomic fragments.

    This is the bridge from the adapter block partition to Vayesta's existing
    fragmentation context.  Fragment ownership follows the block core atoms;
    input/buffer atoms are retained as metadata for the SIESTA backend.
    """

    fragments = []
    for block in blocks:
        core_atoms = list(range(block.core_atom_start, block.core_atom_end))
        input_atoms = list(range(block.input_atom_start, block.input_atom_end))
        buffer_atoms = [atom for atom in input_atoms if atom not in set(core_atoms)]
        fragment = fragmentation.add_atomic_fragment(
            core_atoms,
            orbital_filter=orbital_filter,
            name=f"{name_prefix}-{block.block_id:04d}",
        )
        fragment.siesta_block_id = block.block_id
        fragment.siesta_machine_id = block.machine_id
        fragment.siesta_core_atoms = tuple(core_atoms)
        fragment.siesta_input_atoms = tuple(input_atoms)
        fragment.siesta_buffer_atoms = tuple(buffer_atoms)
        fragment.siesta_core_atom_range = (block.core_atom_start, block.core_atom_end)
        fragment.siesta_input_atom_range = (block.input_atom_start, block.input_atom_end)
        fragments.append(fragment)
    return fragments


def attach_siesta_results_to_fragments(
    fragments: Sequence[object],
    results: Sequence[SiestaEwfResult],
    strict: bool = True,
    predictive_ewf_closure: dict | None = None,
    cluster_hamiltonians: dict | None = None,
    cluster_solver_results: dict | None = None,
    effective_correlated_results: dict | None = None,
) -> list[object]:
    """Attach projected SIESTA EWF results to Vayesta fragments by block id."""

    results_by_block = {result.block_id: result for result in results}
    attached = []
    for fragment in fragments:
        block_id = getattr(fragment, "siesta_block_id", None)
        if block_id is None:
            if strict:
                raise ValueError("Fragment is missing siesta_block_id metadata")
            continue
        result = results_by_block.get(int(block_id))
        if result is None:
            if strict:
                raise ValueError(f"Missing SIESTA EWF result for fragment block {block_id}")
            continue
        fragment.siesta_ewf_result = result
        fragment.siesta_result_attached = True
        fragment.siesta_rank = result.rank
        fragment.siesta_converged = result.converged
        fragment.siesta_total_energy_ev = result.total_energy_ev
        fragment.siesta_density_matrix_path = result.density_matrix_path
        fragment.siesta_hamiltonian_matrix_path = result.hamiltonian_matrix_path
        fragment.siesta_overlap_matrix_path = result.overlap_matrix_path
        fragment.siesta_orbital_index_path = result.orbital_index_path
        fragment.siesta_solver_metadata = dict(result.matrix_metadata.get("elsi", {}))
        fragment.siesta_run_diagnostics = dict(result.run_diagnostics)
        fragment.siesta_core_atom_orbital_ranges = dict(result.core_atom_orbital_ranges)
        fragment.siesta_core_matrix_metadata = dict(result.core_matrix_metadata)
        if predictive_ewf_closure:
            _attach_predictive_ewf_closure_to_fragment(fragment, int(block_id), predictive_ewf_closure)
        if cluster_hamiltonians:
            _attach_cluster_hamiltonian_to_fragment(fragment, int(block_id), cluster_hamiltonians)
        if cluster_solver_results:
            _attach_cluster_solver_result_to_fragment(fragment, int(block_id), cluster_solver_results)
        if effective_correlated_results:
            _attach_effective_correlated_result_to_fragment(fragment, int(block_id), effective_correlated_results)
        attached.append(fragment)
    return attached


def load_siesta_results_to_fragments(
    workdir: str | os.PathLike[str],
    fragments: Sequence[object],
    strict: bool = True,
    require_complete: bool = True,
    require_converged: bool = True,
    require_matrices: bool = True,
) -> list[object]:
    """Project a SIESTA run directory and attach the results to Vayesta fragments."""

    results = project_results_to_ewf(
        workdir,
        require_complete=require_complete,
        require_converged=require_converged,
        require_matrices=require_matrices,
    )
    predictive_closure = _read_optional_json(Path(workdir) / "predictive_ewf_closure.json")
    cluster_hamiltonians = _read_optional_json(Path(workdir) / "cluster_hamiltonians.json")
    cluster_solver_results = _read_optional_json(Path(workdir) / "cluster_solver_results.json")
    effective_correlated_results = _read_optional_json(Path(workdir) / "effective_correlated_results.json")
    return attach_siesta_results_to_fragments(
        fragments,
        results,
        strict=strict,
        predictive_ewf_closure=predictive_closure,
        cluster_hamiltonians=cluster_hamiltonians,
        cluster_solver_results=cluster_solver_results,
        effective_correlated_results=effective_correlated_results,
    )


def read_run_config(environ: dict[str, str] | None = None) -> SiestaRunConfig:
    """Read EWF/SIESTA adapter configuration from environment variables."""

    environ = os.environ if environ is None else environ
    num_machines = _get_int(environ, "EWF_NUM_MACHINES", 1)
    procs_per_machine = _get_int(environ, "EWF_PROCS_PER_MACHINE", 1)
    threads_per_proc = _get_int(environ, "EWF_THREADS_PER_PROC", 1)
    block_atoms = _get_optional_int(environ, "EWF_BLOCK_ATOMS")
    block_groups = _get_optional_int(environ, "EWF_BLOCK_GROUPS")
    group_size_atoms = _get_optional_int(environ, "EWF_GROUP_SIZE_ATOMS")
    solver = SiestaSolverConfig(
        ntpoly_method=_get_int(environ, "EWF_NTPOLY_METHOD", 2),
        ntpoly_filter=_get_positive_float(environ, "EWF_NTPOLY_FILTER", 1.0e-9),
        ntpoly_tolerance=_get_positive_float(environ, "EWF_NTPOLY_TOLERANCE", 1.0e-6),
        max_scf_iterations=_get_int(environ, "EWF_MAX_SCF_ITERATIONS", 150),
        dm_number_pulay=_get_int(environ, "EWF_DM_NUMBER_PULAY", 6),
        dm_mixing_weight=_get_positive_float(environ, "EWF_DM_MIXING_WEIGHT", 0.05),
    )
    return SiestaRunConfig(
        num_machines=num_machines,
        procs_per_machine=procs_per_machine,
        threads_per_proc=threads_per_proc,
        workdir=Path(environ.get("EWF_WORKDIR", "runs")),
        siesta_bin=environ.get("EWF_SIESTA_BIN") or None,
        block_atoms=block_atoms,
        buffer_atoms=_get_nonnegative_int(environ, "EWF_BLOCK_BUFFER_ATOMS", 0),
        block_groups=block_groups,
        group_size_atoms=group_size_atoms,
        buffer_groups=_get_nonnegative_int(environ, "EWF_BLOCK_BUFFER_GROUPS", 0),
        terminal_cap_atoms=_get_nonnegative_int(environ, "EWF_TERMINAL_CAP_ATOMS", 0),
        dry_run=_get_bool(environ, "EWF_SIESTA_DRY_RUN", True),
        predictive_boundary=_get_bool(environ, "EWF_PREDICTIVE_BOUNDARY", False),
        predictive_boundary_damping=_get_positive_float(environ, "EWF_PREDICTIVE_BOUNDARY_DAMPING", 1.0),
        predictive_boundary_rerun=_get_bool(environ, "EWF_PREDICTIVE_BOUNDARY_RERUN", False),
        effective_interaction_u_ev=_get_nonnegative_float(environ, "EWF_EFFECTIVE_INTERACTION_U_EV", 0.0),
        effective_interaction_denominator_shift_ev=_get_positive_float(
            environ,
            "EWF_EFFECTIVE_INTERACTION_DENOMINATOR_SHIFT_EV",
            1.0e-6,
        ),
        solver=solver,
    )


def minimum_buffer_atoms(config: SiestaRunConfig) -> int:
    """Return the atom-count buffer requested by the active partition mode."""

    if config.group_size_atoms is not None:
        return config.buffer_groups * config.group_size_atoms
    return config.buffer_atoms


def run_assigned_blocks(block_dirs: Sequence[Path], config: SiestaRunConfig) -> list[subprocess.CompletedProcess]:
    """Run SIESTA for generated block directories, unless dry-run mode is active."""

    if config.dry_run:
        return []
    if not config.siesta_bin:
        raise ValueError("EWF_SIESTA_BIN is required when EWF_SIESTA_DRY_RUN is false")

    env = _siesta_subprocess_environment(config.threads_per_proc)

    results = []
    for block_dir in block_dirs:
        with (block_dir / "siesta.out").open("w") as stdout:
            result = subprocess.run(
                [config.siesta_bin, "input.fdf"],
                cwd=block_dir,
                env=env,
                stdout=stdout,
                stderr=subprocess.STDOUT,
                check=False,
            )
            results.append(result)
        if result.returncode != 0:
            break
    return results


def read_siesta_output(block_dir: str | os.PathLike[str]) -> SiestaResult:
    """Read minimal SIESTA output data for an EWF block.

    Matrix files such as density, Hamiltonian, and overlap dumps are deliberately
    left as file-level adapter contracts until the exact SIESTA dump settings are
    selected.  This reader extracts the scalar data available in normal text
    output and records convergence status.
    """

    block_dir = Path(block_dir)
    metadata_path = block_dir / "block.json"
    output_path = block_dir / "siesta.out"
    metadata = json.loads(metadata_path.read_text())
    block_id = metadata["block_id"]
    orbital_index_path = _first_existing(block_dir, _ORBITAL_INDEX_CANDIDATES, block_id)
    atom_orbital_ranges = _read_atom_orbital_ranges(orbital_index_path, metadata.get("local_to_global_atom_index", []))
    density_matrix_path = _first_existing(block_dir, _DENSITY_MATRIX_CANDIDATES, block_id)
    hamiltonian_matrix_path = _first_existing(block_dir, _HAMILTONIAN_MATRIX_CANDIDATES, block_id)
    overlap_matrix_path = _first_existing(block_dir, _OVERLAP_MATRIX_CANDIDATES, block_id)
    matrix_metadata = _read_matrix_metadata(density_matrix_path, hamiltonian_matrix_path)
    elsi_metadata = _read_elsi_log_metadata(block_dir / "elsi_log.json")
    if elsi_metadata:
        matrix_metadata["elsi"] = elsi_metadata
    embedding_applied = _read_optional_json(block_dir / "ewf_embedding_potential_applied.json")
    if embedding_applied:
        matrix_metadata["ewf_embedding_potential_applied"] = embedding_applied
    if not output_path.exists():
        return SiestaResult(
            block_id=block_id,
            converged=None,
            total_energy_ev=None,
            wall_time_seconds=None,
            density_matrix_path=density_matrix_path,
            hamiltonian_matrix_path=hamiltonian_matrix_path,
            overlap_matrix_path=overlap_matrix_path,
            orbital_index_path=orbital_index_path,
            atom_orbital_ranges=atom_orbital_ranges,
            output_path=output_path,
            matrix_metadata=matrix_metadata,
            run_diagnostics={},
        )

    text = output_path.read_text(errors="replace")
    converged = _detect_convergence(text)
    total_energy_ev = _detect_total_energy_ev(text)
    wall_time_seconds = _detect_wall_time_seconds(text)
    run_diagnostics = _detect_run_diagnostics(text)
    return SiestaResult(
        block_id=block_id,
        converged=converged,
        total_energy_ev=total_energy_ev,
        wall_time_seconds=wall_time_seconds,
        density_matrix_path=density_matrix_path,
        hamiltonian_matrix_path=hamiltonian_matrix_path,
        overlap_matrix_path=overlap_matrix_path,
        orbital_index_path=orbital_index_path,
        atom_orbital_ranges=atom_orbital_ranges,
        output_path=output_path,
        matrix_metadata=matrix_metadata,
        run_diagnostics=run_diagnostics,
    )


def collect_rank_results(workdir: str | os.PathLike[str]) -> list[dict]:
    """Collect per-rank result manifests into block order."""

    workdir = Path(workdir)
    payload = []
    for result_path in sorted(workdir.glob("result_rank_*.json")):
        rank_text = result_path.stem.rsplit("_", 1)[-1]
        rank_payload = json.loads(result_path.read_text())
        for result in rank_payload:
            result.setdefault("rank", int(rank_text))
            payload.append(result)
    payload.sort(key=lambda result: (result.get("block_id", -1), result.get("rank", -1)))
    return payload


def write_results_manifest(workdir: str | os.PathLike[str]) -> list[dict]:
    """Write `results.json` from all per-rank manifests and return its payload."""

    workdir = Path(workdir)
    payload = collect_rank_results(workdir)
    (workdir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def write_results_snapshot_manifest(workdir: str | os.PathLike[str], label: str) -> dict:
    """Save current result manifests under a stable phase label."""

    workdir = Path(workdir)
    safe_label = label.strip().replace("-", "_")
    results = collect_rank_results(workdir)
    result_path = workdir / f"{safe_label}_results.json"
    result_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    rank_paths = []
    for result_path_in in sorted(workdir.glob("result_rank_*.json")):
        result_path_out = workdir / f"{safe_label}_{result_path_in.name}"
        shutil.copy2(result_path_in, result_path_out)
        rank_paths.append(str(result_path_out))
    payload = {
        "version": 1,
        "label": safe_label,
        "num_results": len(results),
        "results_path": str(result_path),
        "rank_result_paths": rank_paths,
    }
    (workdir / f"{safe_label}_results_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload


def write_embedding_rerun_delta_manifest(workdir: str | os.PathLike[str]) -> dict:
    """Compare first-pass block results with predictive rerun results."""

    workdir = Path(workdir)
    first_path = workdir / "first_pass_results.json"
    rerun_path = workdir / "predictive_rerun_results.json"
    if not first_path.exists():
        raise FileNotFoundError(first_path)
    if not rerun_path.exists():
        raise FileNotFoundError(rerun_path)
    first = {int(item["block_id"]): item for item in json.loads(first_path.read_text())}
    rerun = {int(item["block_id"]): item for item in json.loads(rerun_path.read_text())}
    blocks = []
    for block_id in sorted(set(first) | set(rerun)):
        before = first.get(block_id, {})
        after = rerun.get(block_id, {})
        before_energy = before.get("total_energy_ev")
        after_energy = after.get("total_energy_ev")
        before_wall = before.get("wall_time_seconds")
        after_wall = after.get("wall_time_seconds")
        before_scf = before.get("run_diagnostics", {}).get("num_scf_steps")
        after_scf = after.get("run_diagnostics", {}).get("num_scf_steps")
        applied = after.get("matrix_metadata", {}).get("ewf_embedding_potential_applied", {})
        blocks.append(
            {
                "block_id": block_id,
                "first_pass_converged": before.get("converged"),
                "predictive_rerun_converged": after.get("converged"),
                "first_pass_total_energy_ev": before_energy,
                "predictive_rerun_total_energy_ev": after_energy,
                "delta_total_energy_ev": _difference(after_energy, before_energy),
                "first_pass_wall_time_seconds": before_wall,
                "predictive_rerun_wall_time_seconds": after_wall,
                "delta_wall_time_seconds": _difference(after_wall, before_wall),
                "first_pass_scf_steps": before_scf,
                "predictive_rerun_scf_steps": after_scf,
                "delta_scf_steps": None
                if before_scf is None or after_scf is None
                else int(after_scf) - int(before_scf),
                "embedding_potential_applied": bool(applied),
                "embedding_potential_applied_diagnostics": applied,
            }
        )
    energy_deltas = [block["delta_total_energy_ev"] for block in blocks if block["delta_total_energy_ev"] is not None]
    payload = {
        "version": 1,
        "model": "first-pass-vs-predictive-rerun",
        "num_blocks": len(blocks),
        "all_rerun_blocks_converged": all(block["predictive_rerun_converged"] is True for block in blocks),
        "all_blocks_have_embedding_potential_applied": all(block["embedding_potential_applied"] for block in blocks),
        "total_delta_energy_ev": float(sum(energy_deltas)) if energy_deltas else None,
        "blocks": blocks,
    }
    (workdir / "embedding_rerun_delta.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def summarize_run(workdir: str | os.PathLike[str]) -> dict:
    """Build a weak-scaling-oriented summary from block and result manifests."""

    workdir = Path(workdir)
    blocks = _read_json_list(workdir / "blocks.json")
    results = _read_json_list(workdir / "results.json") if (workdir / "results.json").exists() else collect_rank_results(workdir)
    results_by_block = {int(result["block_id"]): result for result in results}
    wall_times = [
        float(result["wall_time_seconds"])
        for result in results
        if result.get("wall_time_seconds") is not None
    ]
    returncodes = [int(result.get("returncode", 0)) for result in results]
    converged = [result.get("converged") is True for result in results]
    num_blocks = len(blocks)
    num_successful = sum(1 for code in returncodes if code == 0)
    num_failed = sum(1 for code in returncodes if code != 0)
    num_converged = sum(1 for item in converged if item)
    num_unconverged = sum(1 for item in converged if not item)
    schedule_owner = _read_schedule_owner_ranks(workdir)
    block_summaries = [
        _summarize_one_block(
            block,
            results_by_block.get(int(block["block_id"])),
            schedule_owner.get(int(block["block_id"])),
        )
        for block in blocks
    ]
    ranks = sorted({int(result["rank"]) for result in results if result.get("rank") is not None})
    scheduled_ranks = _read_scheduled_ranks(workdir)
    machines = sorted({int(block["machine_id"]) for block in blocks if block.get("machine_id") is not None})
    max_wall_time = None if not wall_times else float(max(wall_times))
    mean_wall_time = None if not wall_times else float(sum(wall_times) / len(wall_times))
    success_rate = _ratio(num_successful, num_blocks)
    converged_rate = _ratio(num_converged, num_blocks)
    return {
        "workdir": str(workdir),
        "num_blocks": num_blocks,
        "num_results": len(results),
        "num_successful_results": num_successful,
        "num_failed_results": num_failed,
        "num_converged_results": num_converged,
        "num_unconverged_results": num_unconverged,
        "success_rate": success_rate,
        "converged_rate": converged_rate,
        "num_scheduled_ranks": len(scheduled_ranks),
        "scheduled_ranks": scheduled_ranks,
        "num_ranks_with_results": len(ranks),
        "ranks_with_results": ranks,
        "num_machines": len(machines),
        "machines": machines,
        "total_wall_time_seconds": None if not wall_times else float(sum(wall_times)),
        "max_block_wall_time_seconds": max_wall_time,
        "max_block_wall_time": max_wall_time,
        "min_block_wall_time_seconds": None if not wall_times else float(min(wall_times)),
        "mean_block_wall_time_seconds": mean_wall_time,
        "mean_block_wall_time": mean_wall_time,
        "solver_used": _summarize_solver_used(results),
        "ntpoly_methods": _summarize_ntpoly_methods(results),
        "max_scf_steps": _summarize_max_scf_steps(results),
        "weak_scaling_efficiency_vs_baseline": _weak_scaling_efficiency(max_wall_time, max_wall_time),
        "blocks": block_summaries,
    }


def write_run_summary_manifest(workdir: str | os.PathLike[str]) -> dict:
    """Write `run_summary.json` with scheduling, success, and timing metrics."""

    workdir = Path(workdir)
    payload = summarize_run(workdir)
    (workdir / "run_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def build_matrix_shape_report(workdir: str | os.PathLike[str]) -> dict:
    """Build per-rank/block local matrix MNK and partition quality diagnostics."""

    workdir = Path(workdir)
    blocks = _read_json_list(workdir / "blocks.json")
    results = _read_json_list(workdir / "results.json") if (workdir / "results.json").exists() else collect_rank_results(workdir)
    ewf_results = _read_json_list(workdir / "ewf_results.json") if (workdir / "ewf_results.json").exists() else []
    schedule = _read_optional_json(workdir / "schedule.json") or {}
    global_matrices = _read_optional_json(workdir / "global_matrices.json") or {}
    results_by_block = {int(result["block_id"]): result for result in results}
    ewf_by_block = {int(result["block_id"]): result for result in ewf_results}
    schedule_owner = _read_schedule_owner_ranks(workdir)
    threads_per_proc = schedule.get("threads_per_proc")

    block_entries = [
        _matrix_shape_block_entry(
            block,
            results_by_block.get(int(block["block_id"])),
            ewf_by_block.get(int(block["block_id"])),
            schedule_owner.get(int(block["block_id"])),
            threads_per_proc,
        )
        for block in blocks
    ]
    rank_entries = _matrix_shape_rank_entries(block_entries, schedule)
    local_norbitals = [entry["local_matrix"]["m"] for entry in block_entries if entry["local_matrix"]["m"] is not None]
    core_norbitals = [entry["core_matrix"]["m"] for entry in block_entries if entry["core_matrix"]["m"] is not None]
    global_norbitals = global_matrices.get("norbitals")
    max_local = max(local_norbitals) if local_norbitals else None
    mean_local = float(sum(local_norbitals) / len(local_norbitals)) if local_norbitals else None
    max_core = max(core_norbitals) if core_norbitals else None
    effective_reduction = None
    if global_norbitals and max_local:
        effective_reduction = float(max_local / global_norbitals)
    balance_ratio = None
    if local_norbitals and mean_local:
        balance_ratio = float(max_local / mean_local)
    return {
        "version": 1,
        "workdir": str(workdir),
        "matrix_shape_model": "square-local-sparse-scf-matrix",
        "mnk_convention": "For each SIESTA block, dense-equivalent M=N=K=local_norbitals. Core MNK uses core-owned orbitals after EWF projection.",
        "num_blocks": len(block_entries),
        "num_ranks": schedule.get("num_ranks"),
        "threads_per_proc": threads_per_proc,
        "global_core_norbitals": global_norbitals,
        "global_core_nnz": global_matrices.get("nnz"),
        "max_local_norbitals": max_local,
        "mean_local_norbitals": mean_local,
        "max_core_norbitals": max_core,
        "local_vs_global_norbital_ratio": effective_reduction,
        "local_balance_ratio_max_over_mean": balance_ratio,
        "effective_partition": _effective_partition_label(effective_reduction, balance_ratio),
        "analysis": _matrix_shape_analysis(effective_reduction, balance_ratio, block_entries),
        "blocks": block_entries,
        "ranks": rank_entries,
    }


def write_matrix_shape_report_manifest(workdir: str | os.PathLike[str]) -> dict:
    """Write `matrix_shape_report.json` with per-process local matrix sizes."""

    workdir = Path(workdir)
    payload = build_matrix_shape_report(workdir)
    (workdir / "matrix_shape_report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def compare_weak_scaling_runs(workdirs: Sequence[str | os.PathLike[str]]) -> dict:
    """Compare multiple `run_summary.json` files for weak-scaling analysis."""

    if not workdirs:
        raise ValueError("At least one workdir is required")
    summaries = [_load_or_create_run_summary(Path(workdir)) for workdir in workdirs]
    baseline_time = _first_nonnull(summary.get("max_block_wall_time_seconds") for summary in summaries)
    runs = [_weak_scaling_run_entry(summary, baseline_time) for summary in summaries]
    current_run = runs[-1]
    return {
        "baseline_workdir": runs[0]["workdir"],
        "baseline_max_block_wall_time_seconds": baseline_time,
        "num_runs": len(runs),
        "current_run": current_run,
        "num_blocks": current_run.get("num_blocks"),
        "scheduled_ranks": current_run.get("scheduled_ranks"),
        "ranks_with_results": current_run.get("ranks_with_results"),
        "success_rate": current_run.get("success_rate"),
        "converged_rate": current_run.get("converged_rate"),
        "max_block_wall_time_seconds": current_run.get("max_block_wall_time_seconds"),
        "max_block_wall_time": current_run.get("max_block_wall_time"),
        "mean_block_wall_time_seconds": current_run.get("mean_block_wall_time_seconds"),
        "mean_block_wall_time": current_run.get("mean_block_wall_time"),
        "solver_used": current_run.get("solver_used", []),
        "ntpoly_methods": current_run.get("ntpoly_methods", []),
        "max_scf_steps": current_run.get("max_scf_steps"),
        "weak_scaling_efficiency_vs_baseline": current_run.get("weak_scaling_efficiency_vs_baseline"),
        "runs": runs,
    }


def write_weak_scaling_report(
    output_path: str | os.PathLike[str],
    workdirs: Sequence[str | os.PathLike[str]],
) -> dict:
    """Write a JSON report comparing multiple EWF/SIESTA runs."""

    output_path = Path(output_path)
    payload = compare_weak_scaling_runs(workdirs)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def project_results_to_ewf(
    workdir: str | os.PathLike[str],
    require_complete: bool = True,
    require_converged: bool = True,
    require_matrices: bool = True,
) -> list[SiestaEwfResult]:
    """Project block results to the core-owned contract consumed by EWF.

    The returned objects keep buffer atoms as context but only expose orbital
    ownership for core atoms.  This avoids treating isolated block/buffer data as
    a final global observable.
    """

    workdir = Path(workdir)
    blocks = _read_json_list(workdir / "blocks.json")
    if (workdir / "results.json").exists():
        results = _read_json_list(workdir / "results.json")
    else:
        results = collect_rank_results(workdir)

    results_by_block = {int(result["block_id"]): result for result in results}
    if require_complete:
        missing = sorted(int(block["block_id"]) for block in blocks if int(block["block_id"]) not in results_by_block)
        if missing:
            raise ValueError(f"Missing SIESTA results for blocks: {missing}")

    projected = []
    for block in blocks:
        block_id = int(block["block_id"])
        result = results_by_block.get(block_id)
        if result is None:
            continue
        if result.get("returncode", 0) != 0:
            raise ValueError(f"SIESTA block {block_id} failed with return code {result.get('returncode')}")
        if require_converged and result.get("converged") is not True:
            raise ValueError(f"SIESTA block {block_id} is not converged")
        if require_matrices:
            missing_paths = [
                key
                for key in ("density_matrix_path", "hamiltonian_matrix_path", "overlap_matrix_path", "orbital_index_path")
                if not result.get(key)
            ]
            if missing_paths:
                raise ValueError(f"SIESTA block {block_id} is missing matrix artifacts: {missing_paths}")
        result = dict(result)
        ewf_result = _project_one_result_to_ewf(block, result)
        if require_matrices:
            missing_core_orbitals = sorted(set(ewf_result.core_atoms) - set(ewf_result.core_atom_orbital_ranges))
            if missing_core_orbitals:
                raise ValueError(f"SIESTA block {block_id} is missing core orbital ranges: {missing_core_orbitals}")
            _validate_matrix_orbital_count(block_id, result)
            result["core_matrix_metadata"] = _read_core_matrix_metadata(
                _str_to_path(result.get("hamiltonian_matrix_path")),
                _str_to_path(result.get("density_matrix_path")),
                ewf_result.core_atom_orbital_ranges,
            )
            ewf_result = _project_one_result_to_ewf(block, result)
        projected.append(ewf_result)
    return projected


def write_ewf_results_manifest(
    workdir: str | os.PathLike[str],
    require_complete: bool = True,
    require_converged: bool = True,
    require_matrices: bool = True,
) -> list[dict]:
    """Write `ewf_results.json` with core-owned block artifacts."""

    workdir = Path(workdir)
    projected = project_results_to_ewf(
        workdir,
        require_complete=require_complete,
        require_converged=require_converged,
        require_matrices=require_matrices,
    )
    payload = [result.to_metadata() for result in projected]
    (workdir / "ewf_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def assemble_global_matrices(
    workdir_or_results: str | os.PathLike[str] | Sequence[SiestaEwfResult],
    natoms: int | None = None,
) -> SiestaGlobalMatrices:
    """Assemble core-owned sparse DM/H/S entries into global orbital numbering.

    This function enforces that each global core atom is owned by exactly one
    block.  It does not perform any embedding boundary correction; it only
    constructs the deterministic, no-double-counting collection surface.
    """

    if isinstance(workdir_or_results, (str, os.PathLike)):
        results = project_results_to_ewf(workdir_or_results)
    else:
        results = list(workdir_or_results)
    _validate_core_atom_coverage(results, natoms=natoms)
    global_ranges = _build_global_orbital_ranges(results)

    rows = []
    cols = []
    density = []
    hamiltonian = []
    overlap = []
    block_ids = []
    nspin_density = nspin_hamiltonian = None

    for result in results:
        local_to_global = _build_local_to_global_orbital_map(
            result.core_atom_orbital_ranges,
            global_ranges,
        )
        dm = result.read_core_density_matrix()
        hsx = result.read_core_hsx_matrix()
        if not np.array_equal(dm.rows, hsx.rows) or not np.array_equal(dm.cols, hsx.cols):
            raise ValueError(f"SIESTA block {result.block_id} DM and HSX sparsity patterns differ")
        if nspin_density is None:
            nspin_density = dm.values.shape[0]
            nspin_hamiltonian = hsx.hamiltonian.shape[0]
        elif nspin_density != dm.values.shape[0] or nspin_hamiltonian != hsx.hamiltonian.shape[0]:
            raise ValueError("Inconsistent spin dimensions across SIESTA blocks")
        rows.append(local_to_global[dm.rows])
        cols.append(local_to_global[dm.cols])
        density.append(dm.values)
        hamiltonian.append(hsx.hamiltonian)
        overlap.append(hsx.overlap)
        block_ids.append(np.full(dm.nnz, result.block_id, dtype=np.int64))

    if rows:
        rows_out = np.concatenate(rows)
        cols_out = np.concatenate(cols)
        density_out = np.concatenate(density, axis=1)
        hamiltonian_out = np.concatenate(hamiltonian, axis=1)
        overlap_out = np.concatenate(overlap)
        block_ids_out = np.concatenate(block_ids)
    else:
        rows_out = cols_out = block_ids_out = np.asarray([], dtype=np.int64)
        density_out = hamiltonian_out = np.zeros((0, 0), dtype=np.float64)
        overlap_out = np.asarray([], dtype=np.float64)
    return SiestaGlobalMatrices(
        atom_orbital_ranges=global_ranges,
        rows=rows_out,
        cols=cols_out,
        density=density_out,
        hamiltonian=hamiltonian_out,
        overlap=overlap_out,
        block_ids=block_ids_out,
    )


def write_global_matrices_manifest(
    workdir: str | os.PathLike[str],
    natoms: int | None = None,
) -> dict:
    """Write `global_matrices.json` with the core-owned assembly summary."""

    workdir = Path(workdir)
    payload = assemble_global_matrices(workdir, natoms=natoms).to_metadata()
    (workdir / "global_matrices.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def build_electron_constraint(
    fdf: FdfInput,
    global_matrices_metadata: dict,
    apply_correction: bool = True,
) -> dict:
    """Build an electron-number closure from valence counts and Tr(D S)."""

    target = estimate_valence_electron_count(fdf)
    observed = global_matrices_metadata.get("density_overlap_trace_total")
    deviation = None if observed is None else float(observed - target)
    correction = None if deviation is None else float(-deviation)
    corrected = None if observed is None or correction is None else float(observed + correction)
    applied = apply_correction and correction is not None
    return {
        "constraint_level": "global-electron-closure" if applied else "diagnostic",
        "target_valence_electrons": float(target),
        "observed_density_overlap_trace": observed,
        "electron_count_deviation": deviation,
        "electron_count_correction": correction if applied else None,
        "corrected_density_overlap_trace": corrected if applied else observed,
        "corrected_electron_count_deviation": 0.0 if applied else deviation,
        "chemical_potential_status": "applied" if applied else "not_applied",
        "policy": "global_trace_shift_closure" if applied else "report_only_no_mu_update",
    }


def write_electron_constraint_manifest(
    workdir: str | os.PathLike[str],
    fdf: FdfInput,
    apply_correction: bool = True,
) -> dict:
    """Write `electron_constraint.json` using `global_matrices.json`."""

    workdir = Path(workdir)
    global_path = workdir / "global_matrices.json"
    if not global_path.exists():
        raise FileNotFoundError(global_path)
    payload = build_electron_constraint(fdf, json.loads(global_path.read_text()), apply_correction=apply_correction)
    (workdir / "electron_constraint.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def build_embedded_observables(workdir: str | os.PathLike[str]) -> dict:
    """Build the minimal embedded-observable manifest from closed diagnostics."""

    workdir = Path(workdir)
    validation = _read_optional_json(workdir / "validation.json") or {}
    corrections = _read_optional_json(workdir / "boundary_corrections.json") or {}
    electron = _read_optional_json(workdir / "electron_constraint.json") or {}
    global_matrices = _read_optional_json(workdir / "global_matrices.json") or {}
    predictive_closure = _read_optional_json(workdir / "predictive_ewf_closure.json") or {}
    cluster_solver = _read_optional_json(workdir / "cluster_solver_results.json") or {}
    effective = _read_optional_json(workdir / "effective_correlated_results.json") or {}
    energy_corrections = [
        float(correction.get("energy_correction_ev", 0.0))
        for correction in corrections.get("corrections", [])
        if correction.get("energy_correction_ev") is not None
    ]
    total_block_energy = validation.get("total_block_energy_ev")
    total_energy = None
    if total_block_energy is not None:
        total_energy = float(total_block_energy) + float(sum(energy_corrections))
    effective_total_energy = None
    if total_energy is not None and effective.get("total_correlation_energy_ev") is not None:
        effective_total_energy = float(total_energy) + float(effective["total_correlation_energy_ev"])
    return {
        "version": 1,
        "observable_level": "minimal-embedded-closure",
        "closure_model": corrections.get("closure_model"),
        "energy_policy": "block_sum_plus_boundary_corrections",
        "total_block_energy_ev": total_block_energy,
        "boundary_energy_correction_ev": float(sum(energy_corrections)),
        "embedded_total_energy_ev": total_energy,
        "electron_policy": electron.get("policy"),
        "target_valence_electrons": electron.get("target_valence_electrons"),
        "observed_density_overlap_trace": electron.get("observed_density_overlap_trace"),
        "corrected_density_overlap_trace": electron.get("corrected_density_overlap_trace"),
        "corrected_electron_count_deviation": electron.get("corrected_electron_count_deviation"),
        "density_overlap_trace_total": global_matrices.get("density_overlap_trace_total"),
        "matrix_ownership": "core_owned",
        "predictive_ewf_closure_level": predictive_closure.get("closure_level"),
        "predictive_ewf_closure_status": predictive_closure.get("status"),
        "predictive_total_energy_ev": (predictive_closure.get("energy") or {}).get("predictive_total_energy_ev"),
        "double_counting_energy_correction_ev": (predictive_closure.get("energy") or {}).get(
            "double_counting_energy_correction_ev"
        ),
        "bath_total_rank": (predictive_closure.get("bath_construction") or {}).get("total_bath_rank"),
        "correlated_solver_status": predictive_closure.get("correlated_solver_status"),
        "cluster_solver_level": cluster_solver.get("solver_level"),
        "cluster_solver_kind": cluster_solver.get("solver_kind"),
        "cluster_solver_ready": cluster_solver.get("ready"),
        "cluster_solver_total_density_projected_one_electron_energy_ev": cluster_solver.get(
            "total_density_projected_one_electron_energy_ev"
        ),
        "cluster_solver_total_aufbau_one_electron_energy_ev": cluster_solver.get(
            "total_aufbau_one_electron_energy_ev"
        ),
        "cluster_solver_correlated_status": cluster_solver.get("correlated_solver_status"),
        "effective_correlated_solver_level": effective.get("solver_level"),
        "effective_correlated_solver_kind": effective.get("solver_kind"),
        "effective_correlated_results_ready": effective.get("ready"),
        "effective_correlation_energy_ev": effective.get("total_correlation_energy_ev"),
        "effective_embedded_total_energy_ev": effective_total_energy,
        "effective_correlated_status": effective.get("correlated_solver_status"),
        "effective_interaction_u_ev": effective.get("effective_interaction_u_ev"),
    }


def write_embedded_observables_manifest(workdir: str | os.PathLike[str]) -> dict:
    """Write `embedded_observables.json` for downstream EWF consumers."""

    workdir = Path(workdir)
    payload = build_embedded_observables(workdir)
    (workdir / "embedded_observables.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def build_embedding_benchmark(
    workdir: str | os.PathLike[str],
    reference_observables: dict,
    energy_tolerance_ev: float = 1.0e-3,
    electron_tolerance: float = 1.0e-6,
) -> dict:
    """Compare closed embedded observables against a reference calculation."""

    workdir = Path(workdir)
    embedded = _read_optional_json(workdir / "embedded_observables.json") or {}
    global_matrices = _read_optional_json(workdir / "global_matrices.json") or {}
    natoms = global_matrices.get("natoms")
    embedded_energy = embedded.get("embedded_total_energy_ev")
    reference_energy = reference_observables.get("total_energy_ev")
    energy_error = _difference(embedded_energy, reference_energy)
    energy_error_per_atom = None
    if energy_error is not None and natoms:
        energy_error_per_atom = float(energy_error / int(natoms))

    embedded_electrons = embedded.get("corrected_density_overlap_trace")
    reference_electrons = reference_observables.get("density_overlap_trace_total")
    electron_error = _difference(embedded_electrons, reference_electrons)
    energy_ok = energy_error is not None and abs(energy_error) <= energy_tolerance_ev
    electron_ok = electron_error is None or abs(electron_error) <= electron_tolerance
    ok = bool(energy_ok and electron_ok)
    return {
        "version": 1,
        "benchmark_level": "embedded_vs_reference_observables",
        "status": "passed" if ok else "failed",
        "reference_label": reference_observables.get("label", "reference"),
        "reference_kind": reference_observables.get("reference_kind", "external_reference"),
        "reference_is_external": bool(reference_observables.get("reference_is_external", True)),
        "natoms": natoms,
        "energy_tolerance_ev": energy_tolerance_ev,
        "electron_tolerance": electron_tolerance,
        "embedded_total_energy_ev": embedded_energy,
        "reference_total_energy_ev": reference_energy,
        "energy_error_ev": energy_error,
        "energy_error_per_atom_ev": energy_error_per_atom,
        "corrected_density_overlap_trace": embedded_electrons,
        "reference_density_overlap_trace": reference_electrons,
        "electron_count_error": electron_error,
        "energy_within_tolerance": energy_ok,
        "electron_count_within_tolerance": electron_ok,
        "ok": ok,
    }


def write_embedding_benchmark_manifest(
    workdir: str | os.PathLike[str],
    reference_observables: dict,
    energy_tolerance_ev: float = 1.0e-3,
    electron_tolerance: float = 1.0e-6,
) -> dict:
    """Write `embedding_benchmark.json` for reference-quality tracking."""

    workdir = Path(workdir)
    payload = build_embedding_benchmark(
        workdir,
        reference_observables,
        energy_tolerance_ev=energy_tolerance_ev,
        electron_tolerance=electron_tolerance,
    )
    (workdir / "embedding_benchmark.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def build_reference_missing_embedding_benchmark(
    workdir: str | os.PathLike[str],
    reason: str,
    next_steps: Sequence[str] = (),
    reference_label: str = "missing-full-system-reference",
) -> dict:
    """Build a benchmark-ready manifest when an external reference is absent."""

    workdir = Path(workdir)
    embedded = _read_optional_json(workdir / "embedded_observables.json") or {}
    global_matrices = _read_optional_json(workdir / "global_matrices.json") or {}
    return {
        "version": 1,
        "benchmark_level": "reference_missing_benchmark_ready_manifest",
        "status": "reference_missing",
        "ok": False,
        "reference_label": reference_label,
        "reference_kind": "missing_external_reference",
        "reference_is_external": False,
        "reference_missing_reason": reason,
        "next_validation_steps": list(next_steps),
        "natoms": global_matrices.get("natoms"),
        "embedded_total_energy_ev": embedded.get("embedded_total_energy_ev"),
        "corrected_density_overlap_trace": embedded.get("corrected_density_overlap_trace"),
    }


def write_reference_missing_embedding_benchmark_manifest(
    workdir: str | os.PathLike[str],
    reason: str,
    next_steps: Sequence[str] = (),
    reference_label: str = "missing-full-system-reference",
) -> dict:
    """Write `embedding_benchmark.json` documenting a missing external reference."""

    workdir = Path(workdir)
    payload = build_reference_missing_embedding_benchmark(
        workdir,
        reason=reason,
        next_steps=next_steps,
        reference_label=reference_label,
    )
    (workdir / "embedding_benchmark.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def reference_observables_from_workdir(
    workdir: str | os.PathLike[str],
    label: str | None = None,
) -> dict:
    """Extract reference observables from a SIESTA/EWF run directory."""

    workdir = Path(workdir)
    embedded = _read_optional_json(workdir / "embedded_observables.json") or {}
    validation = _read_optional_json(workdir / "validation.json") or {}
    global_matrices = _read_optional_json(workdir / "global_matrices.json") or {}
    total_energy = embedded.get("embedded_total_energy_ev", validation.get("total_block_energy_ev"))
    density_trace = embedded.get(
        "corrected_density_overlap_trace",
        global_matrices.get("density_overlap_trace_total"),
    )
    return {
        "label": label or str(workdir),
        "reference_kind": "external_full_system_or_higher_quality_run",
        "reference_is_external": True,
        "total_energy_ev": total_energy,
        "density_overlap_trace_total": density_trace,
        "source_workdir": str(workdir),
    }


def write_embedding_benchmark_from_reference_workdir(
    workdir: str | os.PathLike[str],
    reference_workdir: str | os.PathLike[str],
    label: str | None = None,
    energy_tolerance_ev: float = 1.0e-3,
    electron_tolerance: float = 1.0e-6,
) -> dict:
    """Write a benchmark manifest using another run directory as reference."""

    reference = reference_observables_from_workdir(reference_workdir, label=label)
    return write_embedding_benchmark_manifest(
        workdir,
        reference,
        energy_tolerance_ev=energy_tolerance_ev,
        electron_tolerance=electron_tolerance,
    )


def build_physical_readiness_report(workdir: str | os.PathLike[str]) -> dict:
    """Report whether SIESTA block artifacts are ready for physical EWF use."""

    workdir = Path(workdir)
    validation = _read_optional_json(workdir / "validation.json") or {}
    embedding = _read_optional_json(workdir / "embedding_contract.json") or {}
    corrections = _read_optional_json(workdir / "boundary_corrections.json") or {}
    electron = _read_optional_json(workdir / "electron_constraint.json") or {}
    global_matrices = _read_optional_json(workdir / "global_matrices.json") or {}
    observables = _read_optional_json(workdir / "embedded_observables.json") or {}
    benchmark = _read_optional_json(workdir / "embedding_benchmark.json") or {}
    predictive = _read_optional_json(workdir / "predictive_embedding_potential.json") or {}
    predictive_closure = _read_optional_json(workdir / "predictive_ewf_closure.json") or {}
    clusters = _read_optional_json(workdir / "cluster_hamiltonians.json") or {}
    cluster_solver = _read_optional_json(workdir / "cluster_solver_results.json") or {}
    effective = _read_optional_json(workdir / "effective_correlated_results.json") or {}
    effective_scan = _read_optional_json(workdir / "effective_interaction_benchmark_scan.json") or {}

    backend_ready = bool(validation.get("ok"))
    blockers = []
    if not backend_ready:
        blockers.append("SIESTA backend artifacts failed validation or validation.json is missing")

    unparameterized = int(corrections.get("num_unparameterized_corrections", 0))
    parameterized = int(corrections.get("num_parameterized_corrections", 0))
    pending_embedding = int(embedding.get("num_pending_embedding_terms", 0))
    if pending_embedding and parameterized < pending_embedding:
        blockers.append(f"{pending_embedding - parameterized} boundary embedding terms do not have embedding potentials")
    if unparameterized:
        blockers.append(f"{unparameterized} boundary correction slots are not parameterized")

    if backend_ready and not electron:
        blockers.append("electron_constraint.json is missing")
    elif electron and electron.get("chemical_potential_status") != "applied":
        blockers.append("global electron-number or chemical-potential constraint is not applied")
    if backend_ready and not observables:
        blockers.append("embedded_observables.json is missing")

    embedded_ready = backend_ready and not blockers
    minimal_closure_ready = (
        backend_ready
        and bool(observables)
        and not unparameterized
        and (not electron or electron.get("chemical_potential_status") == "applied")
    )
    reference_calibrated_ready = corrections.get("correction_level") == "reference-calibrated-boundary-closure"
    benchmark_manifest_ready = bool(benchmark)
    reference_benchmark_ready = _benchmark_is_external_reference_ready(benchmark)
    predictive_potential_ready = predictive.get("num_parameterized_terms", 0) > 0 and not predictive.get("uses_reference_energy", True)
    predictive_embedding_ready = bool(
        predictive_potential_ready
        and predictive.get("self_consistency_status") == "converged"
        and predictive.get("sIESTA_external_potential_applied") is True
    )
    predictive_ewf_closure_ready = predictive_closure.get("status") == "ready"
    cluster_hamiltonians_ready = clusters.get("ready") is True
    cluster_solver_ready = cluster_solver.get("ready") is True
    effective_correlated_ready = effective.get("ready") is True
    return {
        "version": 1,
        "backend_artifacts_ready": backend_ready,
        "minimal_embedded_closure_ready": minimal_closure_ready,
        "embedded_observable_ready": embedded_ready,
        "reference_calibrated_correction_ready": reference_calibrated_ready,
        "predictive_boundary_potential_ready": predictive_potential_ready,
        "predictive_ewf_closure_ready": predictive_ewf_closure_ready,
        "cluster_hamiltonians_ready": cluster_hamiltonians_ready,
        "cluster_solver_results_ready": cluster_solver_ready,
        "effective_correlated_results_ready": effective_correlated_ready,
        "effective_interaction_benchmark_scan_ready": bool(effective_scan),
        "production_predictive_physics_ready": bool(predictive_closure.get("production_predictive_physics_ready")),
        "benchmark_manifest_ready": benchmark_manifest_ready,
        "reference_benchmark_ready": reference_benchmark_ready,
        "predictive_embedding_ready": predictive_embedding_ready,
        "predictive_embedding_status": (
            "self_consistent_predictive_embedding_ready"
            if predictive_embedding_ready
            else predictive.get("self_consistency_status", "not_implemented_minimal_or_reference_calibrated_closure_only")
        ),
        "status": "embedded_observable_ready" if embedded_ready else "diagnostic_backend_only",
        "blockers": blockers,
        "diagnostic_outputs": {
            "energy_policy": validation.get("energy_policy"),
            "density_overlap_trace_total": global_matrices.get("density_overlap_trace_total"),
            "electron_count_deviation": electron.get("electron_count_deviation"),
            "corrected_density_overlap_trace": electron.get("corrected_density_overlap_trace"),
            "corrected_electron_count_deviation": electron.get("corrected_electron_count_deviation"),
            "boundary_correction_level": corrections.get("correction_level"),
            "boundary_closure_model": corrections.get("closure_model"),
            "boundary_reference_total_energy_ev": corrections.get("reference_total_energy_ev"),
            "boundary_total_calibrated_energy_correction_ev": corrections.get("total_calibrated_energy_correction_ev"),
            "predictive_potential_level": predictive.get("potential_level"),
            "predictive_potential_model": predictive.get("model"),
            "predictive_potential_uses_reference_energy": predictive.get("uses_reference_energy"),
            "predictive_total_energy_correction_ev": predictive.get("total_predictive_energy_correction_ev"),
            "predictive_siesta_external_potential_applied": predictive.get("sIESTA_external_potential_applied"),
            "predictive_blockers": predictive.get("blockers"),
            "predictive_ewf_closure_level": predictive_closure.get("closure_level"),
            "predictive_ewf_closure_status": predictive_closure.get("status"),
            "predictive_ewf_total_energy_ev": (predictive_closure.get("energy") or {}).get(
                "predictive_total_energy_ev"
            ),
            "predictive_ewf_double_counting_energy_correction_ev": (
                predictive_closure.get("energy") or {}
            ).get("double_counting_energy_correction_ev"),
            "predictive_ewf_bath_total_rank": (predictive_closure.get("bath_construction") or {}).get(
                "total_bath_rank"
            ),
            "predictive_ewf_correlated_solver_status": predictive_closure.get("correlated_solver_status"),
            "production_predictive_blockers": predictive_closure.get("production_blockers"),
            "cluster_hamiltonian_artifact_level": clusters.get("artifact_level"),
            "cluster_hamiltonian_num_written_blocks": clusters.get("num_written_blocks"),
            "cluster_hamiltonian_ready": clusters.get("ready"),
            "cluster_solver_level": cluster_solver.get("solver_level"),
            "cluster_solver_kind": cluster_solver.get("solver_kind"),
            "cluster_solver_ready": cluster_solver.get("ready"),
            "cluster_solver_num_solved_blocks": cluster_solver.get("num_solved_blocks"),
            "cluster_solver_total_density_projected_one_electron_energy_ev": cluster_solver.get(
                "total_density_projected_one_electron_energy_ev"
            ),
            "cluster_solver_total_aufbau_one_electron_energy_ev": cluster_solver.get(
                "total_aufbau_one_electron_energy_ev"
            ),
            "effective_correlated_solver_level": effective.get("solver_level"),
            "effective_correlated_solver_kind": effective.get("solver_kind"),
            "effective_correlated_ready": effective.get("ready"),
            "effective_correlated_status": effective.get("correlated_solver_status"),
            "effective_interaction_u_ev": effective.get("effective_interaction_u_ev"),
            "effective_correlation_energy_ev": effective.get("total_correlation_energy_ev"),
            "effective_uses_ab_initio_two_electron_integrals": effective.get(
                "uses_ab_initio_two_electron_integrals"
            ),
            "effective_interaction_scan_status": effective_scan.get("status"),
            "effective_interaction_scan_best_u_ev": (effective_scan.get("best_sample") or {}).get(
                "effective_interaction_u_ev"
            ),
            "effective_interaction_scan_best_energy_error_ev": (effective_scan.get("best_sample") or {}).get(
                "energy_error_ev"
            ),
            "effective_interaction_reference_fit_possible": effective_scan.get(
                "reference_fit_possible_with_real_nonnegative_u"
            ),
            "effective_interaction_reference_fit_reason": effective_scan.get("reference_fit_reason"),
            "embedded_total_energy_ev": observables.get("embedded_total_energy_ev"),
            "benchmark_ok": benchmark.get("ok"),
            "benchmark_status": benchmark.get("status"),
            "benchmark_reference_kind": benchmark.get("reference_kind"),
            "benchmark_reference_is_external": benchmark.get("reference_is_external"),
            "benchmark_reference_missing_reason": benchmark.get("reference_missing_reason"),
            "benchmark_energy_error_ev": benchmark.get("energy_error_ev"),
            "benchmark_energy_error_per_atom_ev": benchmark.get("energy_error_per_atom_ev"),
        },
    }


def write_physical_readiness_manifest(workdir: str | os.PathLike[str]) -> dict:
    """Write `physical_readiness.json` for downstream EWF consumers."""

    workdir = Path(workdir)
    payload = build_physical_readiness_report(workdir)
    (workdir / "physical_readiness.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def estimate_valence_electron_count(fdf: FdfInput) -> int:
    """Estimate total valence electrons from FDF species atomic numbers."""

    return sum(_valence_electrons(fdf.species_atomic_numbers.get(atom.species), fdf.species_labels.get(atom.species)) for atom in fdf.atoms)


def validate_ewf_results(
    workdir_or_results: str | os.PathLike[str] | Sequence[SiestaEwfResult],
    natoms: int | None = None,
    min_buffer_atoms: int = 0,
    require_complete: bool = True,
    require_converged: bool = True,
    require_matrices: bool = True,
) -> SiestaValidationReport:
    """Validate the current SIESTA block collection contract.

    This report is intentionally strict about engineering invariants and
    intentionally conservative about physics.  It can prove that the adapter has
    a complete no-double-counting core-owned collection, but it does not claim a
    final embedded total energy.
    """

    errors: list[str] = []
    warnings: list[str] = []
    boundary_errors = _read_boundary_errors(workdir_or_results) if isinstance(workdir_or_results, (str, os.PathLike)) else []
    embedding_errors, embedding_warnings = (
        _read_embedding_contract_messages(workdir_or_results)
        if isinstance(workdir_or_results, (str, os.PathLike))
        else ([], [])
    )
    electron_warnings = _read_electron_constraint_warnings(workdir_or_results) if isinstance(workdir_or_results, (str, os.PathLike)) else []
    correction_warnings = _read_boundary_correction_warnings(workdir_or_results) if isinstance(workdir_or_results, (str, os.PathLike)) else []
    closure_state = _read_closure_state(workdir_or_results) if isinstance(workdir_or_results, (str, os.PathLike)) else {}
    try:
        if isinstance(workdir_or_results, (str, os.PathLike)):
            results = project_results_to_ewf(
                workdir_or_results,
                require_complete=require_complete,
                require_converged=require_converged,
                require_matrices=require_matrices,
            )
        else:
            results = list(workdir_or_results)
        _validate_core_atom_coverage(results, natoms=natoms)
        global_matrices = assemble_global_matrices(results, natoms=natoms) if require_matrices else None
    except Exception as exc:
        return SiestaValidationReport(
            ok=False,
            errors=(str(exc),),
            warnings=(),
            natoms=natoms,
            nblocks=0,
            ncore_atoms=0,
            norbitals=0,
            nnz=0,
            density_overlap_trace_total=None,
            density_overlap_trace_by_spin=(),
            total_block_energy_ev=None,
            energy_policy="invalid",
        )
    errors.extend(boundary_errors)
    errors.extend(embedding_errors)
    warnings.extend(embedding_warnings)
    warnings.extend(electron_warnings)
    warnings.extend(correction_warnings)

    seen_core_atoms = {atom for result in results for atom in result.core_atoms}
    if natoms is not None and len(seen_core_atoms) != natoms:
        errors.append(f"Core atom coverage has {len(seen_core_atoms)} atoms, expected {natoms}")
    for result in results:
        if result.converged is not True:
            errors.append(f"Block {result.block_id} is not converged")
        if result.density_matrix_path is None:
            errors.append(f"Block {result.block_id} has no density matrix")
        if result.hamiltonian_matrix_path is None or result.overlap_matrix_path is None:
            errors.append(f"Block {result.block_id} has no HSX Hamiltonian/overlap matrix")
        elsi_metadata = result.matrix_metadata.get("elsi", {})
        solver_used = {str(solver).upper() for solver in elsi_metadata.get("solver_used", [])}
        if elsi_metadata and solver_used != {"NTPOLY"}:
            errors.append(f"Block {result.block_id} used ELSI solver {sorted(solver_used)}, expected ['NTPOLY']")
        if elsi_metadata and elsi_metadata.get("complete") is False:
            warnings.append(f"Block {result.block_id} has an incomplete ELSI log; solver metadata was read from complete records only")
        nt_method = elsi_metadata.get("last_solver_settings", {}).get("nt_method")
        if nt_method is not None and int(nt_method) != 2:
            errors.append(f"Block {result.block_id} used NTPoly method {nt_method}, expected TRS2 method 2")
        if min_buffer_atoms > 0 and _is_internal_block(result, natoms) and len(result.buffer_atoms) < min_buffer_atoms:
            warnings.append(
                f"Block {result.block_id} has {len(result.buffer_atoms)} buffer atoms; "
                f"requested at least {min_buffer_atoms}"
            )

    total_energy = _sum_block_energies(results)
    if total_energy is not None and not closure_state.get("embedded_observables"):
        warnings.append(
            "total_block_energy_ev is a diagnostic sum of independent block energies; "
            "it is not an embedded total energy."
        )
    if not closure_state.get("boundary_corrections_applied") or not closure_state.get("electron_constraint_applied"):
        warnings.append(
            "Current matrices are core-owned SIESTA block collections without boundary embedding potential, "
            "chemical-potential constraint, or double-counting energy correction."
        )
    if global_matrices is not None and not closure_state.get("electron_constraint_applied"):
        warnings.append(
            "density_overlap_trace_total is Tr(D S) over the core-owned assembled sparse matrices; "
            "it is a diagnostic electron-count proxy, not an enforced EWF electron-number constraint."
        )
    return SiestaValidationReport(
        ok=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        natoms=natoms,
        nblocks=len(results),
        ncore_atoms=len(seen_core_atoms),
        norbitals=0 if global_matrices is None else global_matrices.norbitals,
        nnz=0 if global_matrices is None else global_matrices.nnz,
        density_overlap_trace_total=None if global_matrices is None else global_matrices.density_overlap_trace_total,
        density_overlap_trace_by_spin=() if global_matrices is None else global_matrices.density_overlap_trace_by_spin,
        total_block_energy_ev=total_energy,
        energy_policy=(
            "minimal_embedded_closure"
            if closure_state.get("embedded_observables")
            else "diagnostic_block_sum_not_embedded_total"
            if total_energy is not None
            else "not_available"
        ),
    )


def write_validation_manifest(
    workdir: str | os.PathLike[str],
    natoms: int | None = None,
    min_buffer_atoms: int = 0,
    require_complete: bool = True,
    require_converged: bool = True,
    require_matrices: bool = True,
) -> dict:
    """Write `validation.json` with EWF/SIESTA adapter diagnostics."""

    workdir = Path(workdir)
    payload = validate_ewf_results(
        workdir,
        natoms=natoms,
        min_buffer_atoms=min_buffer_atoms,
        require_complete=require_complete,
        require_converged=require_converged,
        require_matrices=require_matrices,
    ).to_metadata()
    (workdir / "validation.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def default_block_atoms(natoms: int, num_machines: int, procs_per_machine: int) -> int:
    """Choose a conservative default block size from the simulated topology."""

    nworkers = max(1, num_machines * procs_per_machine)
    return max(1, math.ceil(natoms / nworkers))


def _render_block_fdf(
    fdf: FdfInput,
    atoms: Sequence[Atom],
    block_id: int,
    solver_config: SiestaSolverConfig | None = None,
) -> str:
    out: list[str] = []
    natoms = len(atoms)
    solver_config = SiestaSolverConfig() if solver_config is None else solver_config
    forced_options = solver_config.to_fdf_options()
    seen_options: set[str] = set()
    for lineno, line in enumerate(fdf.lines):
        lower = line.strip().lower()
        if lineno <= fdf.coordinates_start or lineno >= fdf.coordinates_end:
            if lower.startswith("systemlabel"):
                out.append(f"SystemLabel      block_{block_id:04d}")
            elif lower.startswith("numberofatoms"):
                out.append(f"NumberOfAtoms    {natoms}")
            elif _option_key(lower) in forced_options:
                key = _option_key(lower)
                seen_options.add(key)
                out.append(forced_options[key])
            else:
                out.append(line)
        elif lineno == fdf.coordinates_start + 1:
            for atom in atoms:
                out.append(f"    {atom.x:16.9f} {atom.y:16.9f} {atom.z:16.9f}    {atom.species}")
    missing_options = [value for key, value in forced_options.items() if key not in seen_options]
    if missing_options:
        out.extend(["", "# Required solver and block outputs for the EWF/SIESTA adapter.", *missing_options])
    return "\n".join(out) + "\n"


def _project_one_result_to_ewf(block: dict, result: dict) -> SiestaEwfResult:
    core_start = int(block["core_atom_start"])
    core_end = int(block["core_atom_end"])
    input_atoms = [int(atom) for atom in block.get("local_to_global_atom_index", [])]
    core_atoms = tuple(atom for atom in input_atoms if core_start <= atom < core_end)
    buffer_atoms = tuple(atom for atom in input_atoms if atom < core_start or atom >= core_end)
    atom_orbital_ranges = {
        int(atom): (int(bounds[0]), int(bounds[1]))
        for atom, bounds in result.get("atom_orbital_ranges", {}).items()
    }
    core_atom_orbital_ranges = {
        atom: atom_orbital_ranges[atom]
        for atom in core_atoms
        if atom in atom_orbital_ranges
    }
    return SiestaEwfResult(
        block_id=int(block["block_id"]),
        machine_id=block.get("machine_id"),
        rank=result.get("rank"),
        core_atom_range=(core_start, core_end),
        input_atom_range=(int(block["input_atom_start"]), int(block["input_atom_end"])),
        core_atoms=core_atoms,
        buffer_atoms=buffer_atoms,
        core_atom_orbital_ranges=core_atom_orbital_ranges,
        converged=result.get("converged"),
        total_energy_ev=result.get("total_energy_ev"),
        density_matrix_path=_str_to_path(result.get("density_matrix_path")),
        hamiltonian_matrix_path=_str_to_path(result.get("hamiltonian_matrix_path")),
        overlap_matrix_path=_str_to_path(result.get("overlap_matrix_path")),
        orbital_index_path=_str_to_path(result.get("orbital_index_path")),
        output_path=_str_to_path(result.get("output_path")),
        matrix_metadata=result.get("matrix_metadata", {}),
        core_matrix_metadata=result.get("core_matrix_metadata", {}),
        run_diagnostics=result.get("run_diagnostics", {}),
    )


def _validate_core_atom_coverage(results: Sequence[SiestaEwfResult], natoms: int | None = None) -> None:
    seen: dict[int, int] = {}
    duplicates = {}
    for result in results:
        for atom in result.core_atoms:
            if atom in seen:
                duplicates.setdefault(atom, [seen[atom]]).append(result.block_id)
            seen[atom] = result.block_id
    if duplicates:
        raise ValueError(f"Duplicate core atom ownership: {duplicates}")
    if natoms is not None:
        missing = sorted(set(range(natoms)) - set(seen))
        if missing:
            raise ValueError(f"Missing core atom ownership: {missing}")


def _is_internal_block(result: SiestaEwfResult, natoms: int | None) -> bool:
    if natoms is None:
        return result.core_atom_range[0] > result.input_atom_range[0] and result.core_atom_range[1] < result.input_atom_range[1]
    return result.core_atom_range[0] > 0 and result.core_atom_range[1] < natoms


def _sum_block_energies(results: Sequence[SiestaEwfResult]) -> float | None:
    energies = [result.total_energy_ev for result in results]
    if not energies or any(energy is None for energy in energies):
        return None
    return float(sum(energies))


def _read_boundary_errors(workdir_or_results) -> list[str]:
    workdir = Path(workdir_or_results)
    boundary_path = workdir / "boundary.json"
    if not boundary_path.exists():
        return []
    payload = json.loads(boundary_path.read_text())
    uncovered = payload.get("uncovered_boundary_bonds", [])
    if not uncovered:
        return []
    return [
        f"Block {bond['block_id']} boundary bond {bond['atom_i']}-{bond['atom_j']} is not covered by input buffer"
        for bond in uncovered
    ]


def _read_embedding_contract_messages(workdir_or_results) -> tuple[list[str], list[str]]:
    workdir = Path(workdir_or_results)
    contract_path = workdir / "embedding_contract.json"
    if not contract_path.exists():
        return [], []
    payload = json.loads(contract_path.read_text())
    corrections = _read_optional_json(workdir / "boundary_corrections.json") or {}
    parameterized = int(corrections.get("num_parameterized_corrections", 0))
    errors = [
        f"Block {term['block_id']} embedding term {term['bond_atoms']} has uncovered boundary"
        for term in payload.get("terms", [])
        if term.get("status") == "invalid_uncovered_boundary"
    ]
    pending = int(payload.get("num_pending_embedding_terms", 0))
    unresolved = max(0, pending - parameterized)
    warnings = []
    if unresolved:
        warnings.append(
            f"{unresolved} boundary embedding terms require embedding potential and energy correction; "
            "current adapter records the contract but does not apply those corrections."
        )
    return errors, warnings


def _atom_distance(atom_i: Atom, atom_j: Atom) -> float:
    return math.sqrt(
        (atom_i.x - atom_j.x) ** 2
        + (atom_i.y - atom_j.y) ** 2
        + (atom_i.z - atom_j.z) ** 2
    )


def _covalent_radius(fdf: FdfInput, species: int) -> float:
    label = fdf.species_labels.get(species, "").strip().capitalize()
    radii = {
        "H": 0.31,
        "C": 0.76,
        "N": 0.71,
        "O": 0.66,
        "S": 1.05,
        "P": 1.07,
    }
    if label in radii:
        return radii[label]
    fallback = {1: 0.76, 2: 0.31}
    return fallback.get(species, 0.8)


def _valence_electrons(atomic_number: int | None, label: str | None) -> int:
    if atomic_number is not None:
        table = {
            1: 1,
            5: 3,
            6: 4,
            7: 5,
            8: 6,
            9: 7,
            14: 4,
            15: 5,
            16: 6,
            17: 7,
        }
        if atomic_number in table:
            return table[atomic_number]
    label_table = {
        "H": 1,
        "B": 3,
        "C": 4,
        "N": 5,
        "O": 6,
        "F": 7,
        "Si": 4,
        "P": 5,
        "S": 6,
        "Cl": 7,
    }
    if label:
        normalized = label.strip().capitalize()
        if normalized in label_table:
            return label_table[normalized]
    return 0


def _read_electron_constraint_warnings(workdir_or_results) -> list[str]:
    workdir = Path(workdir_or_results)
    path = workdir / "electron_constraint.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    if payload.get("chemical_potential_status") == "applied":
        return []
    deviation = payload.get("electron_count_deviation")
    if deviation is None:
        return []
    return [
        "electron_constraint.json reports diagnostic electron-count deviation "
        f"{deviation}; chemical-potential correction is not applied."
    ]


def _read_boundary_correction_warnings(workdir_or_results) -> list[str]:
    workdir = Path(workdir_or_results)
    path = workdir / "boundary_corrections.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    count = int(payload.get("num_unparameterized_corrections", 0))
    if not count:
        return []
    return [
        f"{count} boundary correction slots are not parameterized; "
        "Hamiltonian embedding potentials and energy corrections are not applied."
    ]


def _read_closure_state(workdir_or_results) -> dict[str, bool]:
    workdir = Path(workdir_or_results)
    corrections = _read_optional_json(workdir / "boundary_corrections.json") or {}
    electron = _read_optional_json(workdir / "electron_constraint.json") or {}
    observables = _read_optional_json(workdir / "embedded_observables.json") or {}
    return {
        "boundary_corrections_applied": bool(corrections)
        and int(corrections.get("num_unparameterized_corrections", 0)) == 0,
        "electron_constraint_applied": electron.get("chemical_potential_status") == "applied",
        "embedded_observables": bool(observables),
    }


def _attach_predictive_ewf_closure_to_fragment(fragment: object, block_id: int, closure: dict) -> None:
    bath_terms = [
        dict(term)
        for term in (closure.get("bath_construction") or {}).get("terms", [])
        if int(term.get("block_id", -1)) == block_id
    ]
    potential_terms = [
        dict(term)
        for term in (closure.get("double_counting") or {}).get("terms", [])
        if int(term.get("block_id", -1)) == block_id
    ]
    fragment.siesta_predictive_ewf_closure = {
        "version": closure.get("version"),
        "closure_level": closure.get("closure_level"),
        "status": closure.get("status"),
        "uses_reference_energy": closure.get("uses_reference_energy"),
        "production_predictive_physics_ready": closure.get("production_predictive_physics_ready"),
        "correlated_solver_status": closure.get("correlated_solver_status"),
        "energy": dict(closure.get("energy") or {}),
        "production_blockers": list(closure.get("production_blockers") or []),
    }
    fragment.siesta_predictive_bath_terms = bath_terms
    fragment.siesta_embedding_potential_expectation_terms = potential_terms
    fragment.siesta_predictive_ewf_closure_ready = closure.get("status") == "ready"
    fragment.siesta_production_predictive_physics_ready = bool(closure.get("production_predictive_physics_ready"))
    fragment.siesta_predictive_total_energy_ev = (closure.get("energy") or {}).get("predictive_total_energy_ev")
    fragment.siesta_predictive_double_counting_energy_ev = (closure.get("energy") or {}).get(
        "double_counting_energy_correction_ev"
    )


def _attach_cluster_hamiltonian_to_fragment(fragment: object, block_id: int, clusters: dict) -> None:
    block_metadata = next(
        (dict(block) for block in clusters.get("blocks", []) if int(block.get("block_id", -1)) == block_id),
        None,
    )
    fragment.siesta_cluster_hamiltonians_ready = clusters.get("ready") is True
    fragment.siesta_cluster_hamiltonian_manifest = {
        "version": clusters.get("version"),
        "artifact_level": clusters.get("artifact_level"),
        "basis_model": clusters.get("basis_model"),
        "orthogonalization": clusters.get("orthogonalization"),
        "ready": clusters.get("ready"),
    }
    fragment.siesta_cluster_hamiltonian_metadata = block_metadata
    fragment.siesta_cluster_hamiltonian_path = None if block_metadata is None else Path(block_metadata["npz_path"])
    fragment.siesta_cluster_ready_for_correlated_solver = bool(
        block_metadata and block_metadata.get("ready_for_correlated_solver")
    )


def _attach_cluster_solver_result_to_fragment(fragment: object, block_id: int, solver_results: dict) -> None:
    block_result = next(
        (dict(block) for block in solver_results.get("blocks", []) if int(block.get("block_id", -1)) == block_id),
        None,
    )
    fragment.siesta_cluster_solver_results_ready = solver_results.get("ready") is True
    fragment.siesta_cluster_solver_manifest = {
        "version": solver_results.get("version"),
        "solver_level": solver_results.get("solver_level"),
        "solver_kind": solver_results.get("solver_kind"),
        "ready": solver_results.get("ready"),
        "correlated_solver_status": solver_results.get("correlated_solver_status"),
    }
    fragment.siesta_cluster_solver_result = block_result
    fragment.siesta_cluster_solver_status = None if block_result is None else block_result.get("solver_status")
    fragment.siesta_cluster_one_electron_energy_ev = None if block_result is None else block_result.get(
        "density_projected_one_electron_energy_ev"
    )
    fragment.siesta_cluster_aufbau_energy_ev = None if block_result is None else block_result.get(
        "aufbau_one_electron_energy_ev"
    )


def _attach_effective_correlated_result_to_fragment(fragment: object, block_id: int, payload: dict) -> None:
    block_result = next(
        (dict(block) for block in payload.get("blocks", []) if int(block.get("block_id", -1)) == block_id),
        None,
    )
    fragment.siesta_effective_correlated_results_ready = payload.get("ready") is True
    fragment.siesta_effective_correlated_manifest = {
        "version": payload.get("version"),
        "solver_level": payload.get("solver_level"),
        "solver_kind": payload.get("solver_kind"),
        "ready": payload.get("ready"),
        "correlated_solver_status": payload.get("correlated_solver_status"),
        "uses_ab_initio_two_electron_integrals": payload.get("uses_ab_initio_two_electron_integrals"),
        "effective_interaction_u_ev": payload.get("effective_interaction_u_ev"),
    }
    fragment.siesta_effective_correlated_result = block_result
    fragment.siesta_effective_correlation_energy_ev = None if block_result is None else block_result.get(
        "correlation_energy_ev"
    )
    fragment.siesta_effective_correlated_solver_status = None if block_result is None else block_result.get(
        "solver_status"
    )


def _difference(value, reference) -> float | None:
    if value is None or reference is None:
        return None
    return float(value) - float(reference)


def _summarize_one_block(block: dict, result: dict | None, scheduled_rank: int | None = None) -> dict:
    core_start = int(block["core_atom_start"])
    core_end = int(block["core_atom_end"])
    input_start = int(block["input_atom_start"])
    input_end = int(block["input_atom_end"])
    summary = {
        "block_id": int(block["block_id"]),
        "machine_id": block.get("machine_id"),
        "rank": scheduled_rank if result is None else result.get("rank", scheduled_rank),
        "core_atoms": core_end - core_start,
        "input_atoms": input_end - input_start,
        "buffer_atoms": (core_start - input_start) + (input_end - core_end),
        "returncode": None if result is None else result.get("returncode"),
        "converged": None if result is None else result.get("converged"),
        "wall_time_seconds": None if result is None else result.get("wall_time_seconds"),
        "total_energy_ev": None if result is None else result.get("total_energy_ev"),
        "density_norbitals": None,
        "hamiltonian_norbitals": None,
        "solver_used": None,
        "ntpoly_method": None,
        "ntpoly_filter": None,
        "ntpoly_tolerance": None,
        "num_scf_steps": None,
        "last_scf_step": None,
        "last_scf_energy_ev": None,
        "convergence_reason": None,
    }
    if result is not None:
        matrix_metadata = result.get("matrix_metadata", {})
        density = matrix_metadata.get("density", {})
        hsx = matrix_metadata.get("hamiltonian_overlap", {})
        elsi = matrix_metadata.get("elsi", {})
        solver_used = elsi.get("solver_used", [])
        last_solver_settings = elsi.get("last_solver_settings", {})
        run_diagnostics = result.get("run_diagnostics", {})
        summary["density_norbitals"] = density.get("norbitals")
        summary["hamiltonian_norbitals"] = hsx.get("norbitals")
        summary["solver_used"] = solver_used
        summary["ntpoly_method"] = last_solver_settings.get("nt_method")
        summary["ntpoly_filter"] = last_solver_settings.get("nt_filter")
        summary["ntpoly_tolerance"] = last_solver_settings.get("nt_tol")
        summary["num_scf_steps"] = run_diagnostics.get("num_scf_steps")
        summary["last_scf_step"] = run_diagnostics.get("last_scf_step")
        summary["last_scf_energy_ev"] = run_diagnostics.get("last_scf_energy_ev")
        summary["convergence_reason"] = run_diagnostics.get("convergence_reason")
    return summary


def _summarize_solver_used(results: Sequence[dict]) -> list[str]:
    solvers = set()
    for result in results:
        solvers.update(str(solver).upper() for solver in result.get("matrix_metadata", {}).get("elsi", {}).get("solver_used", []))
    return sorted(solvers)


def _summarize_ntpoly_methods(results: Sequence[dict]) -> list[int]:
    methods = set()
    for result in results:
        method = result.get("matrix_metadata", {}).get("elsi", {}).get("last_solver_settings", {}).get("nt_method")
        if method is not None:
            methods.add(int(method))
    return sorted(methods)


def _summarize_max_scf_steps(results: Sequence[dict]) -> int | None:
    steps = [
        int(result.get("run_diagnostics", {}).get("num_scf_steps"))
        for result in results
        if result.get("run_diagnostics", {}).get("num_scf_steps") is not None
    ]
    return None if not steps else max(steps)


def _matrix_shape_block_entry(
    block: dict,
    result: dict | None,
    ewf_result: dict | None,
    scheduled_rank: int | None,
    threads_per_proc: int | None,
) -> dict:
    core_start = int(block["core_atom_start"])
    core_end = int(block["core_atom_end"])
    input_start = int(block["input_atom_start"])
    input_end = int(block["input_atom_end"])
    rank = scheduled_rank if result is None else result.get("rank", scheduled_rank)
    matrix_metadata = {} if result is None else result.get("matrix_metadata", {})
    density = matrix_metadata.get("density", {})
    hsx = matrix_metadata.get("hamiltonian_overlap", {})
    local_norbitals = _first_int(hsx.get("norbitals"), density.get("norbitals"))
    local_nnz = _first_int(hsx.get("nnz"), density.get("nnz"))
    if local_nnz is None and result is not None:
        local_nnz = _read_local_matrix_nnz(result)
    core_metadata = {} if ewf_result is None else ewf_result.get("core_matrix_metadata", {})
    core_density = core_metadata.get("density", {})
    core_hsx = core_metadata.get("hamiltonian_overlap", {})
    core_norbitals = _first_int(core_hsx.get("norbitals"), core_density.get("norbitals"))
    core_nnz = _first_int(core_hsx.get("nnz"), core_density.get("nnz"))
    return {
        "block_id": int(block["block_id"]),
        "rank": rank,
        "machine_id": block.get("machine_id"),
        "threads_per_proc": threads_per_proc,
        "core_atom_range": [core_start, core_end],
        "input_atom_range": [input_start, input_end],
        "core_atoms": core_end - core_start,
        "input_atoms": input_end - input_start,
        "buffer_atoms": (core_start - input_start) + (input_end - core_end),
        "local_matrix": _square_mnk(local_norbitals, local_nnz),
        "core_matrix": _square_mnk(core_norbitals, core_nnz),
        "buffer_orbital_amplification": _ratio(local_norbitals, core_norbitals),
        "returncode": None if result is None else result.get("returncode"),
        "converged": None if result is None else result.get("converged"),
        "wall_time_seconds": None if result is None else result.get("wall_time_seconds"),
        "solver_used": [] if result is None else matrix_metadata.get("elsi", {}).get("solver_used", []),
        "ntpoly_method": None if result is None else matrix_metadata.get("elsi", {}).get("last_solver_settings", {}).get("nt_method"),
        "num_scf_steps": None if result is None else result.get("run_diagnostics", {}).get("num_scf_steps"),
    }


def _matrix_shape_rank_entries(block_entries: Sequence[dict], schedule: dict) -> list[dict]:
    ranks = {
        int(rank_info["rank"]): {
            "rank": int(rank_info["rank"]),
            "machine_id": rank_info.get("machine_id"),
            "local_rank": rank_info.get("local_rank"),
            "scheduled_block_ids": list(rank_info.get("block_ids", [])),
            "completed_block_ids": [],
            "num_scheduled_blocks": int(rank_info.get("num_blocks", 0)),
            "num_completed_blocks": 0,
            "max_local_m": None,
            "max_local_n": None,
            "max_local_k": None,
            "sum_dense_equivalent_gemm_flops": 0,
            "sum_local_sparse_nnz": 0,
        }
        for rank_info in schedule.get("ranks", [])
    }
    for entry in block_entries:
        rank = entry.get("rank")
        if rank is None:
            continue
        rank = int(rank)
        ranks.setdefault(
            rank,
            {
                "rank": rank,
                "machine_id": entry.get("machine_id"),
                "local_rank": None,
                "scheduled_block_ids": [],
                "completed_block_ids": [],
                "num_scheduled_blocks": 0,
                "num_completed_blocks": 0,
                "max_local_m": None,
                "max_local_n": None,
                "max_local_k": None,
                "sum_dense_equivalent_gemm_flops": 0,
                "sum_local_sparse_nnz": 0,
            },
        )
        rank_entry = ranks[rank]
        rank_entry["completed_block_ids"].append(entry["block_id"])
        rank_entry["num_completed_blocks"] += 1
        local = entry["local_matrix"]
        for axis, key in (("m", "max_local_m"), ("n", "max_local_n"), ("k", "max_local_k")):
            value = local.get(axis)
            if value is not None:
                rank_entry[key] = value if rank_entry[key] is None else max(rank_entry[key], value)
        if local.get("dense_equivalent_gemm_flops") is not None:
            rank_entry["sum_dense_equivalent_gemm_flops"] += int(local["dense_equivalent_gemm_flops"])
        if local.get("nnz") is not None:
            rank_entry["sum_local_sparse_nnz"] += int(local["nnz"])
    return [ranks[key] for key in sorted(ranks)]


def _read_local_matrix_nnz(result: dict) -> int | None:
    for key, reader in (
        ("hamiltonian_matrix_path", read_hsx_sparse),
        ("density_matrix_path", read_density_matrix_sparse),
    ):
        path = _str_to_path(result.get(key))
        if path is None or not path.exists():
            continue
        try:
            return int(reader(path).nnz)
        except Exception:
            continue
    return None


def _nspin_hamiltonian_from_result(result: dict) -> int:
    metadata = result.get("matrix_metadata", {}).get("hamiltonian_overlap", {})
    value = metadata.get("nspin")
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def _ensure_fdf_option(path: Path, key: str, value: str) -> None:
    lines = path.read_text().splitlines()
    option = f"{key}   {value}"
    key_lower = key.lower()
    replaced = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if _option_key(stripped.lower()) == key_lower:
            lines[index] = option
            replaced = True
            break
    if not replaced:
        lines.extend(["", "# EWF predictive embedding potential.", option])
    path.write_text("\n".join(lines) + "\n")


def _siesta_embedding_potential_applied(workdir: Path) -> bool:
    blocks = _read_optional_json(workdir / "blocks.json") or []
    if not blocks:
        return False
    applied = []
    for block in blocks:
        output = workdir / f"block_{int(block['block_id']):04d}" / "siesta.out"
        if not output.exists():
            applied.append(False)
            continue
        text = output.read_text(errors="replace")
        applied.append("ewf_embedding_potential: applied" in text)
    return bool(applied) and all(applied)


def _compute_boundary_coupling_from_result(result: dict, core_atom: int, environment_atom: int) -> dict:
    ranges = {
        int(atom): (int(bounds[0]), int(bounds[1]))
        for atom, bounds in result.get("atom_orbital_ranges", {}).items()
    }
    if core_atom not in ranges:
        raise ValueError(f"missing core atom orbital range {core_atom}")
    if environment_atom not in ranges:
        raise ValueError(f"missing environment atom orbital range {environment_atom}")
    hsx_path = _str_to_path(result.get("hamiltonian_matrix_path"))
    dm_path = _str_to_path(result.get("density_matrix_path"))
    if hsx_path is None or not hsx_path.exists():
        raise ValueError("missing HSX matrix")
    if dm_path is None or not dm_path.exists():
        raise ValueError("missing density matrix")
    hsx = read_hsx_sparse(hsx_path)
    dm = read_density_matrix_sparse(dm_path)
    core_range = ranges[core_atom]
    env_range = ranges[environment_atom]
    core_mask_h = (hsx.rows >= core_range[0]) & (hsx.rows < core_range[1])
    core_mask_h |= (hsx.cols >= core_range[0]) & (hsx.cols < core_range[1])
    env_mask_h = (hsx.rows >= env_range[0]) & (hsx.rows < env_range[1])
    env_mask_h |= (hsx.cols >= env_range[0]) & (hsx.cols < env_range[1])
    crossing = core_mask_h & env_mask_h
    if not np.any(crossing):
        return {
            "num_core_orbitals": core_range[1] - core_range[0],
            "num_environment_orbitals": env_range[1] - env_range[0],
            "num_coupling_entries": 0,
            "density_hamiltonian_coupling_ev": 0.0,
            "density_overlap_population": 0.0,
            "sparse_hamiltonian_embedding_entries": [],
        }
    dm_lookup = {
        (int(row), int(col)): index
        for index, (row, col) in enumerate(zip(dm.rows.tolist(), dm.cols.tolist()))
    }
    coupling = 0.0
    overlap_population = 0.0
    num_entries = 0
    sparse_entries = []
    for hsx_index in np.nonzero(crossing)[0]:
        key = (int(hsx.rows[hsx_index]), int(hsx.cols[hsx_index]))
        dm_index = dm_lookup.get(key)
        if dm_index is None:
            continue
        nspin = min(dm.density.shape[0], hsx.hamiltonian.shape[0])
        if nspin:
            coupling += float(np.dot(dm.density[:nspin, dm_index], hsx.hamiltonian[:nspin, hsx_index]))
            overlap_population += float(np.sum(dm.density[:nspin, dm_index]) * hsx.overlap[hsx_index])
            for spin in range(nspin):
                density_value = float(dm.density[spin, dm_index])
                hamiltonian_value = float(hsx.hamiltonian[spin, hsx_index])
                if density_value == 0.0 and hamiltonian_value == 0.0:
                    continue
                sparse_entries.append(
                    {
                        "row": int(hsx.rows[hsx_index]),
                        "col": int(hsx.cols[hsx_index]),
                        "spin": int(spin),
                        "density": density_value,
                        "source_hamiltonian_ev": hamiltonian_value,
                        "value_ev": float(-0.5 * hamiltonian_value),
                    }
                )
            num_entries += 1
    return {
        "num_core_orbitals": core_range[1] - core_range[0],
        "num_environment_orbitals": env_range[1] - env_range[0],
        "num_coupling_entries": num_entries,
        "density_hamiltonian_coupling_ev": float(coupling),
        "density_overlap_population": float(overlap_population),
        "sparse_hamiltonian_embedding_entries": sparse_entries,
    }


def _compute_boundary_bath_from_result(
    result: dict,
    core_atom: int,
    environment_atom: int,
    threshold: float = 1.0e-6,
) -> dict:
    ranges = {
        int(atom): (int(bounds[0]), int(bounds[1]))
        for atom, bounds in result.get("atom_orbital_ranges", {}).items()
    }
    if core_atom not in ranges:
        raise ValueError(f"core atom {core_atom} has no orbital range")
    if environment_atom not in ranges:
        raise ValueError(f"environment atom {environment_atom} has no orbital range")
    dm_path = result.get("density_matrix_path")
    if not dm_path:
        raise ValueError("density_matrix_path is missing")
    dm = read_density_matrix_sparse(dm_path)
    core_range = ranges[core_atom]
    env_range = ranges[environment_atom]
    ncore = core_range[1] - core_range[0]
    nenv = env_range[1] - env_range[0]
    block = np.zeros((ncore, nenv), dtype=float)
    for index, (row, col) in enumerate(zip(dm.rows.tolist(), dm.cols.tolist())):
        row = int(row)
        col = int(col)
        value = float(np.sum(dm.density[:, index]))
        if core_range[0] <= row < core_range[1] and env_range[0] <= col < env_range[1]:
            block[row - core_range[0], col - env_range[0]] += value
        elif env_range[0] <= row < env_range[1] and core_range[0] <= col < core_range[1]:
            block[col - core_range[0], row - env_range[0]] += value
    singular_values = np.linalg.svd(block, compute_uv=False) if block.size else np.array([], dtype=float)
    bath_rank = int(np.count_nonzero(singular_values > float(threshold)))
    frobenius_norm = float(np.linalg.norm(block))
    retained_norm = float(np.linalg.norm(singular_values[:bath_rank])) if bath_rank else 0.0
    return {
        "block_id": int(result["block_id"]),
        "core_atom": int(core_atom),
        "environment_atom": int(environment_atom),
        "core_orbitals": int(ncore),
        "environment_orbitals": int(nenv),
        "model": "boundary-density-svd-v1",
        "threshold": float(threshold),
        "bath_rank": bath_rank,
        "singular_values": [float(value) for value in singular_values.tolist()],
        "frobenius_norm": frobenius_norm,
        "retained_norm": retained_norm,
        "discarded_norm": float(max(0.0, frobenius_norm * frobenius_norm - retained_norm * retained_norm) ** 0.5),
    }


def _compute_embedding_potential_expectation(result: dict, potential_path: Path) -> dict:
    dm_path = result.get("density_matrix_path")
    if not dm_path:
        raise ValueError("density_matrix_path is missing")
    dm = read_density_matrix_sparse(dm_path)
    dm_lookup = {
        (int(row), int(col)): index
        for index, (row, col) in enumerate(zip(dm.rows.tolist(), dm.cols.tolist()))
    }
    expectation = 0.0
    matched = 0
    missing = 0
    num_entries = 0
    for line in potential_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) < 4:
            continue
        row = int(fields[0]) - 1
        col = int(fields[1]) - 1
        spin = int(fields[2]) - 1
        value = float(fields[3])
        num_entries += 1
        dm_index = dm_lookup.get((row, col))
        if dm_index is None:
            dm_index = dm_lookup.get((col, row))
        if dm_index is None or spin < 0 or spin >= dm.density.shape[0]:
            missing += 1
            continue
        expectation += float(dm.density[spin, dm_index]) * value
        matched += 1
    return {
        "block_id": int(result["block_id"]),
        "potential_file": str(potential_path),
        "present": True,
        "num_entries": num_entries,
        "matched_entries": matched,
        "missing_density_entries": missing,
        "embedding_potential_expectation_ev": float(expectation),
    }


def _write_cluster_hamiltonian_for_block(
    workdir: Path,
    block: dict,
    result: dict,
    closure: dict,
    bath_threshold: float,
    overlap_eigenvalue_floor: float,
) -> dict:
    block_id = int(block["block_id"])
    hsx_path = result.get("hamiltonian_matrix_path")
    dm_path = result.get("density_matrix_path")
    if not hsx_path:
        raise ValueError("hamiltonian_matrix_path is missing")
    if not dm_path:
        raise ValueError("density_matrix_path is missing")
    hsx = read_hsx_sparse(hsx_path)
    dm = read_density_matrix_sparse(dm_path)
    h_dense, s_dense = _dense_hsx_matrices(hsx)
    d_dense = _dense_density_matrices(dm)
    atom_ranges = {
        int(atom): (int(bounds[0]), int(bounds[1]))
        for atom, bounds in result.get("atom_orbital_ranges", {}).items()
    }
    local_to_global = [int(atom) for atom in block.get("local_to_global_atom_index", [])]
    core_global = set(range(int(block["core_atom_start"]), int(block["core_atom_end"])))
    core_local_atoms = [
        local_atom
        for local_atom, global_atom in enumerate(local_to_global)
        if global_atom in core_global and local_atom in atom_ranges
    ]
    environment_local_atoms = [
        local_atom
        for local_atom, global_atom in enumerate(local_to_global)
        if global_atom not in core_global and local_atom in atom_ranges
    ]
    core_indices = _orbital_indices_for_atoms(atom_ranges, core_local_atoms)
    environment_indices = _orbital_indices_for_atoms(atom_ranges, environment_local_atoms)
    bath_vectors, bath_terms = _cluster_bath_vectors_from_closure(
        result,
        closure,
        core_indices,
        environment_indices,
        bath_threshold=bath_threshold,
    )
    basis, basis_labels = _cluster_basis_matrix(hsx.metadata.norbitals, core_indices, bath_vectors)
    s_cluster = _symmetrize_dense(basis.T @ s_dense @ basis)
    h_cluster = np.asarray([_symmetrize_dense(basis.T @ h_spin @ basis) for h_spin in h_dense])
    d_cluster = np.asarray([_symmetrize_dense(basis.T @ s_dense @ d_spin @ s_dense @ basis) for d_spin in d_dense])
    eigvals, eigvecs = np.linalg.eigh(s_cluster)
    active = eigvals > float(overlap_eigenvalue_floor)
    if not np.any(active):
        raise ValueError("cluster overlap has no eigenvalues above floor")
    lowdin = eigvecs[:, active] @ np.diag(1.0 / np.sqrt(eigvals[active]))
    h_orth = np.asarray([_symmetrize_dense(lowdin.T @ h_spin @ lowdin) for h_spin in h_cluster])
    d_orth = np.asarray([_symmetrize_dense(lowdin.T @ d_spin @ lowdin) for d_spin in d_cluster])
    s_orth = _symmetrize_dense(lowdin.T @ s_cluster @ lowdin)
    block_dir = workdir / f"block_{block_id:04d}"
    npz_path = block_dir / f"cluster_hamiltonian_block_{block_id:04d}.npz"
    np.savez_compressed(
        npz_path,
        basis_coefficients=basis,
        overlap_cluster=s_cluster,
        hamiltonian_cluster=h_cluster,
        density_cluster=d_cluster,
        lowdin_orthogonalizer=lowdin,
        overlap_orthogonalized=s_orth,
        hamiltonian_orthogonalized=h_orth,
        density_orthogonalized=d_orth,
        overlap_eigenvalues=eigvals,
        core_orbital_indices=np.asarray(core_indices, dtype=np.int64),
        environment_orbital_indices=np.asarray(environment_indices, dtype=np.int64),
    )
    metadata = {
        "version": 1,
        "block_id": block_id,
        "artifact_level": "solver-ready-one-electron-cluster-v1",
        "npz_path": str(npz_path),
        "source_hamiltonian_matrix_path": str(hsx_path),
        "source_density_matrix_path": str(dm_path),
        "basis_model": "core-ao-plus-boundary-density-svd-bath",
        "orthogonalization": "lowdin",
        "nspin": int(hsx.metadata.nspin),
        "source_norbitals": int(hsx.metadata.norbitals),
        "num_core_orbitals": len(core_indices),
        "num_environment_orbitals": len(environment_indices),
        "num_bath_orbitals": len(bath_vectors),
        "cluster_basis_size": int(basis.shape[1]),
        "orthogonalized_basis_size": int(lowdin.shape[1]),
        "overlap_min_eigenvalue": float(np.min(eigvals)) if eigvals.size else None,
        "overlap_max_eigenvalue": float(np.max(eigvals)) if eigvals.size else None,
        "overlap_eigenvalue_floor": float(overlap_eigenvalue_floor),
        "num_discarded_overlap_vectors": int(np.count_nonzero(~active)),
        "basis_labels": basis_labels,
        "core_local_atoms": core_local_atoms,
        "environment_local_atoms": environment_local_atoms,
        "bath_terms": bath_terms,
        "ready_for_correlated_solver": True,
        "correlated_solver_status": "cluster_hamiltonian_ready_solver_not_run",
    }
    json_path = block_dir / f"cluster_hamiltonian_block_{block_id:04d}.json"
    json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def _dense_hsx_matrices(hsx: SiestaHsxMatrix) -> tuple[np.ndarray, np.ndarray]:
    norb = int(hsx.metadata.norbitals)
    nspin = int(hsx.metadata.nspin)
    h_dense = np.zeros((nspin, norb, norb), dtype=float)
    s_dense = np.zeros((norb, norb), dtype=float)
    for index, (row, col) in enumerate(zip(hsx.rows.tolist(), hsx.cols.tolist())):
        row = int(row)
        col = int(col)
        s_dense[row, col] += float(hsx.overlap[index])
        for spin in range(nspin):
            h_dense[spin, row, col] += float(hsx.hamiltonian[spin, index])
    return np.asarray([_symmetrize_dense(mat) for mat in h_dense]), _symmetrize_dense(s_dense)


def _dense_density_matrices(dm: SiestaDensityMatrix) -> np.ndarray:
    norb = int(dm.metadata.norbitals)
    nspin = int(dm.metadata.nspin)
    d_dense = np.zeros((nspin, norb, norb), dtype=float)
    for index, (row, col) in enumerate(zip(dm.rows.tolist(), dm.cols.tolist())):
        row = int(row)
        col = int(col)
        for spin in range(nspin):
            d_dense[spin, row, col] += float(dm.density[spin, index])
    return np.asarray([_symmetrize_dense(mat) for mat in d_dense])


def _symmetrize_dense(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)


def _orbital_indices_for_atoms(atom_ranges: dict[int, tuple[int, int]], atoms: Sequence[int]) -> list[int]:
    return [
        orbital
        for atom in atoms
        for orbital in range(int(atom_ranges[atom][0]), int(atom_ranges[atom][1]))
    ]


def _cluster_bath_vectors_from_closure(
    result: dict,
    closure: dict,
    core_indices: Sequence[int],
    environment_indices: Sequence[int],
    bath_threshold: float,
) -> tuple[list[np.ndarray], list[dict]]:
    block_id = int(result["block_id"])
    atom_ranges = {
        int(atom): (int(bounds[0]), int(bounds[1]))
        for atom, bounds in result.get("atom_orbital_ranges", {}).items()
    }
    dm = read_density_matrix_sparse(result["density_matrix_path"])
    d_dense = np.sum(_dense_density_matrices(dm), axis=0)
    bath_vectors: list[np.ndarray] = []
    bath_terms = []
    for term in (closure.get("bath_construction") or {}).get("terms", []):
        if int(term.get("block_id", -1)) != block_id:
            continue
        core_atom = int(term["core_atom"])
        env_atom = int(term["environment_atom"])
        if core_atom not in atom_ranges or env_atom not in atom_ranges:
            continue
        core_range = atom_ranges[core_atom]
        env_range = atom_ranges[env_atom]
        core = list(range(core_range[0], core_range[1]))
        env = list(range(env_range[0], env_range[1]))
        density_block = d_dense[np.ix_(core, env)]
        if density_block.size == 0:
            continue
        _u, singular_values, vh = np.linalg.svd(density_block, full_matrices=False)
        rank = int(np.count_nonzero(singular_values > float(bath_threshold)))
        term_vectors = []
        for vector_index in range(rank):
            coeff = np.zeros(dm.metadata.norbitals, dtype=float)
            coeff[env] = vh[vector_index, :]
            bath_vectors.append(coeff)
            term_vectors.append(len(bath_vectors) - 1)
        bath_terms.append(
            {
                "core_atom": core_atom,
                "environment_atom": env_atom,
                "singular_values": [float(value) for value in singular_values.tolist()],
                "bath_rank": rank,
                "bath_vector_indices": term_vectors,
            }
        )
    if not bath_vectors and environment_indices:
        for orbital in environment_indices:
            coeff = np.zeros(dm.metadata.norbitals, dtype=float)
            coeff[int(orbital)] = 1.0
            bath_vectors.append(coeff)
        bath_terms.append(
            {
                "fallback": "environment-ao-unit-vectors",
                "bath_rank": len(environment_indices),
                "bath_vector_indices": list(range(len(environment_indices))),
            }
        )
    return bath_vectors, bath_terms


def _cluster_basis_matrix(
    norbitals: int,
    core_indices: Sequence[int],
    bath_vectors: Sequence[np.ndarray],
) -> tuple[np.ndarray, list[dict]]:
    columns = []
    labels = []
    for orbital in core_indices:
        coeff = np.zeros(int(norbitals), dtype=float)
        coeff[int(orbital)] = 1.0
        labels.append({"kind": "core_ao", "source_orbital": int(orbital)})
        columns.append(coeff)
    for index, vector in enumerate(bath_vectors):
        labels.append({"kind": "bath_svd", "bath_vector": int(index)})
        columns.append(np.asarray(vector, dtype=float))
    if not columns:
        raise ValueError("cluster basis has no core or bath vectors")
    return np.column_stack(columns), labels


def _solve_one_cluster_hamiltonian(block: dict) -> dict:
    npz_path = Path(block["npz_path"])
    arrays = np.load(npz_path)
    h_orth = np.asarray(arrays["hamiltonian_orthogonalized"], dtype=float)
    d_orth = np.asarray(arrays["density_orthogonalized"], dtype=float)
    if h_orth.ndim != 3:
        raise ValueError("hamiltonian_orthogonalized must have shape (nspin, norb, norb)")
    if d_orth.shape != h_orth.shape:
        raise ValueError("density_orthogonalized shape does not match hamiltonian_orthogonalized")
    nspin = int(h_orth.shape[0])
    max_occupation = 2.0 if nspin == 1 else 1.0
    spin_results = []
    density_energy = 0.0
    aufbau_energy = 0.0
    total_electrons = 0.0
    for spin in range(nspin):
        h_spin = _symmetrize_dense(h_orth[spin])
        d_spin = _symmetrize_dense(d_orth[spin])
        eigenvalues = np.linalg.eigvalsh(h_spin)
        electron_count = float(np.trace(d_spin))
        occupations = _aufbau_occupations(eigenvalues.size, electron_count, max_occupation)
        spin_density_energy = float(np.trace(d_spin @ h_spin))
        spin_aufbau_energy = float(np.dot(occupations, eigenvalues))
        density_energy += spin_density_energy
        aufbau_energy += spin_aufbau_energy
        total_electrons += electron_count
        spin_results.append(
            {
                "spin": spin,
                "num_orbitals": int(eigenvalues.size),
                "electron_count_from_density": electron_count,
                "max_occupation": max_occupation,
                "num_fractional_occupations": int(np.count_nonzero((occupations > 0.0) & (occupations < max_occupation))),
                "eigenvalue_min_ev": float(eigenvalues[0]) if eigenvalues.size else None,
                "eigenvalue_max_ev": float(eigenvalues[-1]) if eigenvalues.size else None,
                "eigenvalue_sum_ev": float(np.sum(eigenvalues)),
                "occupied_eigenvalue_energy_ev": spin_aufbau_energy,
                "density_projected_energy_ev": spin_density_energy,
                "first_eigenvalues_ev": [float(value) for value in eigenvalues[: min(8, eigenvalues.size)].tolist()],
            }
        )
    return {
        "block_id": int(block["block_id"]),
        "cluster_npz_path": str(npz_path),
        "solver_status": "solved",
        "solver_kind": "one_electron_reference",
        "correlated_solver_status": "not_run_one_electron_reference_only",
        "production_predictive_physics_ready": False,
        "cluster_basis_size": int(block["cluster_basis_size"]),
        "orthogonalized_basis_size": int(block["orthogonalized_basis_size"]),
        "num_core_orbitals": int(block["num_core_orbitals"]),
        "num_bath_orbitals": int(block["num_bath_orbitals"]),
        "electron_count_from_density": total_electrons,
        "density_projected_one_electron_energy_ev": float(density_energy),
        "aufbau_one_electron_energy_ev": float(aufbau_energy),
        "density_vs_aufbau_energy_delta_ev": float(density_energy - aufbau_energy),
        "spin_channels": spin_results,
    }


def _aufbau_occupations(norb: int, electron_count: float, max_occupation: float) -> np.ndarray:
    remaining = max(0.0, float(electron_count))
    occupations = np.zeros(int(norb), dtype=float)
    for index in range(int(norb)):
        if remaining <= 0.0:
            break
        occ = min(float(max_occupation), remaining)
        occupations[index] = occ
        remaining -= occ
    return occupations


def _solve_effective_interaction_cluster_block(
    block: dict,
    effective_interaction_u_ev: float,
    denominator_shift_ev: float,
) -> dict:
    npz_path = Path(block["npz_path"])
    arrays = np.load(npz_path)
    h_orth = np.asarray(arrays["hamiltonian_orthogonalized"], dtype=float)
    d_orth = np.asarray(arrays["density_orthogonalized"], dtype=float)
    if h_orth.ndim != 3 or d_orth.shape != h_orth.shape:
        raise ValueError("cluster Hamiltonian and density arrays must have matching (nspin, norb, norb) shape")
    nspin = int(h_orth.shape[0])
    max_occupation = 2.0 if nspin == 1 else 1.0
    total_corr = 0.0
    spin_channels = []
    for spin in range(nspin):
        h_spin = _symmetrize_dense(h_orth[spin])
        d_spin = _symmetrize_dense(d_orth[spin])
        eigenvalues, eigenvectors = np.linalg.eigh(h_spin)
        electron_count = float(np.trace(d_spin))
        occupations = _aufbau_occupations(eigenvalues.size, electron_count, max_occupation)
        occupation_fractions = occupations / max_occupation if max_occupation else occupations
        pair_terms = []
        spin_corr = 0.0
        for occ_index, occ_fraction in enumerate(occupation_fractions):
            if occ_fraction <= 0.0:
                continue
            for virt_index, virt_fraction_occupied in enumerate(occupation_fractions):
                virt_fraction = 1.0 - float(virt_fraction_occupied)
                if virt_index <= occ_index or virt_fraction <= 0.0:
                    continue
                denominator = float(eigenvalues[virt_index] - eigenvalues[occ_index] + denominator_shift_ev)
                if denominator <= 0.0:
                    continue
                coupling = _effective_interaction_coupling(
                    eigenvectors[:, occ_index],
                    eigenvectors[:, virt_index],
                    effective_interaction_u_ev,
                )
                energy = -float(occ_fraction) * virt_fraction * coupling * coupling / denominator
                spin_corr += energy
                if len(pair_terms) < 32:
                    pair_terms.append(
                        {
                            "occupied_orbital": int(occ_index),
                            "virtual_orbital": int(virt_index),
                            "occupation_fraction": float(occ_fraction),
                            "virtual_fraction": float(virt_fraction),
                            "denominator_ev": denominator,
                            "coupling_ev": coupling,
                            "energy_contribution_ev": energy,
                        }
                    )
        total_corr += spin_corr
        spin_channels.append(
            {
                "spin": spin,
                "electron_count_from_density": electron_count,
                "num_orbitals": int(eigenvalues.size),
                "num_pair_terms_sampled": len(pair_terms),
                "correlation_energy_ev": float(spin_corr),
                "pair_terms_sample": pair_terms,
            }
        )
    return {
        "block_id": int(block["block_id"]),
        "cluster_npz_path": str(npz_path),
        "solver_status": "solved",
        "solver_kind": "model_correlated_effective_interaction",
        "interaction_model": "local-hubbard-like-lowdin-orbital-overlap",
        "uses_ab_initio_two_electron_integrals": False,
        "effective_interaction_u_ev": float(effective_interaction_u_ev),
        "denominator_shift_ev": float(denominator_shift_ev),
        "correlation_energy_ev": float(total_corr),
        "cluster_basis_size": int(block["cluster_basis_size"]),
        "orthogonalized_basis_size": int(block["orthogonalized_basis_size"]),
        "spin_channels": spin_channels,
    }


def _effective_interaction_coupling(occupied_vector: np.ndarray, virtual_vector: np.ndarray, u_ev: float) -> float:
    return float(u_ev) * float(np.sum((np.asarray(occupied_vector) ** 2) * (np.asarray(virtual_vector) ** 2)))


def _predictive_boundary_error_term(term: dict, reason: str) -> dict:
    return {
        "block_id": int(term["block_id"]),
        "bond_atoms": list(term.get("bond_atoms", [])),
        "core_atom": term.get("core_atom"),
        "environment_atom": term.get("environment_atom"),
        "status": "not_parameterized",
        "error": reason,
    }


def _square_mnk(norbitals: int | None, nnz: int | None = None) -> dict:
    if norbitals is None:
        return {
            "m": None,
            "n": None,
            "k": None,
            "nnz": nnz,
            "dense_equivalent_gemm_flops": None,
            "sparse_fill_fraction": None,
        }
    dense_entries = int(norbitals) * int(norbitals)
    return {
        "m": int(norbitals),
        "n": int(norbitals),
        "k": int(norbitals),
        "nnz": nnz,
        "dense_equivalent_gemm_flops": int(2 * int(norbitals) * int(norbitals) * int(norbitals)),
        "sparse_fill_fraction": None if nnz is None or dense_entries == 0 else float(nnz / dense_entries),
    }


def _effective_partition_label(local_vs_global_ratio: float | None, balance_ratio: float | None) -> str:
    if local_vs_global_ratio is None:
        return "insufficient_global_reference"
    if local_vs_global_ratio < 0.75 and (balance_ratio is None or balance_ratio <= 1.2):
        return "effective_and_balanced"
    if local_vs_global_ratio < 0.75:
        return "effective_but_imbalanced"
    if balance_ratio is not None and balance_ratio > 1.2:
        return "weak_or_imbalanced"
    return "weak_partition_reduction"


def _matrix_shape_analysis(local_vs_global_ratio: float | None, balance_ratio: float | None, block_entries: Sequence[dict]) -> dict:
    amplification = [
        float(entry["buffer_orbital_amplification"])
        for entry in block_entries
        if entry.get("buffer_orbital_amplification") is not None
    ]
    return {
        "local_matrix_reduction_present": None if local_vs_global_ratio is None else local_vs_global_ratio < 1.0,
        "strong_local_reduction": None if local_vs_global_ratio is None else local_vs_global_ratio < 0.75,
        "rank_balance_good": None if balance_ratio is None else balance_ratio <= 1.2,
        "max_buffer_orbital_amplification": None if not amplification else max(amplification),
        "mean_buffer_orbital_amplification": None if not amplification else float(sum(amplification) / len(amplification)),
        "interpretation": (
            "Global core norbital count is unavailable; compare per-block M/N/K only."
            if local_vs_global_ratio is None
            else "Each SIESTA rank sees a smaller square matrix than the assembled global core matrix."
            if local_vs_global_ratio < 1.0
            else "The largest local SIESTA matrix is not smaller than the assembled global core matrix."
        ),
    }


def _first_int(*values) -> int | None:
    for value in values:
        if value is not None:
            return int(value)
    return None


def _load_or_create_run_summary(workdir: Path) -> dict:
    summary_path = workdir / "run_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    return summarize_run(workdir)


def _weak_scaling_run_entry(summary: dict, baseline_time: float | None) -> dict:
    max_time = summary.get("max_block_wall_time_seconds")
    mean_time = summary.get("mean_block_wall_time_seconds")
    success_rate = summary.get("success_rate", _ratio(summary.get("num_successful_results"), summary.get("num_blocks")))
    converged_rate = summary.get("converged_rate", _ratio(summary.get("num_converged_results"), summary.get("num_blocks")))
    return {
        "workdir": summary.get("workdir"),
        "num_blocks": summary.get("num_blocks"),
        "num_scheduled_ranks": summary.get("num_scheduled_ranks"),
        "scheduled_ranks": summary.get("scheduled_ranks", []),
        "num_ranks_with_results": summary.get("num_ranks_with_results"),
        "ranks_with_results": summary.get("ranks_with_results", []),
        "num_machines": summary.get("num_machines"),
        "num_successful_results": summary.get("num_successful_results"),
        "num_failed_results": summary.get("num_failed_results"),
        "success_rate": success_rate,
        "converged_rate": converged_rate,
        "max_block_wall_time_seconds": max_time,
        "max_block_wall_time": max_time,
        "mean_block_wall_time_seconds": mean_time,
        "mean_block_wall_time": mean_time,
        "solver_used": summary.get("solver_used", []),
        "ntpoly_methods": summary.get("ntpoly_methods", []),
        "max_scf_steps": summary.get("max_scf_steps"),
        "weak_scaling_efficiency_vs_baseline": _weak_scaling_efficiency(baseline_time, max_time),
    }


def _weak_scaling_efficiency(baseline_time: float | None, current_time: float | None) -> float | None:
    if baseline_time is None or current_time is None or current_time <= 0:
        return None
    return float(baseline_time / current_time)


def _benchmark_is_external_reference_ready(benchmark: dict) -> bool:
    if not benchmark:
        return False
    if benchmark.get("status") == "reference_missing":
        return False
    if benchmark.get("reference_is_external") is False:
        return False
    label = str(benchmark.get("reference_label", "")).lower()
    if "self" in label or "smoke" in label:
        return False
    return bool(benchmark.get("ok"))


def _ratio(numerator, denominator) -> float | None:
    if denominator in (None, 0):
        return None
    return float((numerator or 0) / denominator)


def _first_nonnull(values: Iterable[float | None]) -> float | None:
    for value in values:
        if value is not None:
            return float(value)
    return None


def _schedule_block_metadata(block: SiestaBlock) -> dict:
    return {
        "block_id": block.block_id,
        "machine_id": block.machine_id,
        "core_atom_start": block.core_atom_start,
        "core_atom_end": block.core_atom_end,
        "input_atom_start": block.input_atom_start,
        "input_atom_end": block.input_atom_end,
        "core_atoms": block.core_atom_end - block.core_atom_start,
        "input_atoms": block.input_atom_end - block.input_atom_start,
        "buffer_atoms": block.buffer_left + block.buffer_right,
    }


def _read_scheduled_ranks(workdir: Path) -> list[int]:
    schedule_path = workdir / "schedule.json"
    if not schedule_path.exists():
        return []
    payload = json.loads(schedule_path.read_text())
    return [
        int(rank_info["rank"])
        for rank_info in payload.get("ranks", [])
        if int(rank_info.get("num_blocks", 0)) > 0
    ]


def _read_schedule_owner_ranks(workdir: Path) -> dict[int, int]:
    schedule_path = workdir / "schedule.json"
    if not schedule_path.exists():
        return {}
    payload = json.loads(schedule_path.read_text())
    return {
        int(block_id): int(rank)
        for block_id, rank in payload.get("block_owner_rank", {}).items()
    }


def _siesta_subprocess_environment(threads_per_proc: int) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not _is_mpi_launcher_environment(key)
    }
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[key] = str(threads_per_proc)
    return env


def _is_mpi_launcher_environment(name: str) -> bool:
    prefixes = (
        "OMPI_",
        "PMI_",
        "PMIX_",
        "MPI_LOCAL",
        "MPI_UNIVERSE",
        "HYDI_",
        "I_MPI_",
    )
    return name.startswith(prefixes)


def _build_global_orbital_ranges(results: Sequence[SiestaEwfResult]) -> dict[int, tuple[int, int]]:
    ranges = {}
    cursor = 0
    for result in sorted(results, key=lambda item: item.core_atom_range):
        for atom in result.core_atoms:
            if atom not in result.core_atom_orbital_ranges:
                raise ValueError(f"SIESTA block {result.block_id} has no orbital range for core atom {atom}")
            local_start, local_end = result.core_atom_orbital_ranges[atom]
            width = local_end - local_start
            if width <= 0:
                raise ValueError(f"SIESTA block {result.block_id} has invalid orbital range for atom {atom}")
            ranges[atom] = (cursor, cursor + width)
            cursor += width
    return ranges


def _build_local_to_global_orbital_map(
    local_ranges: dict[int, tuple[int, int]],
    global_ranges: dict[int, tuple[int, int]],
) -> np.ndarray:
    local_norb = max(end for _, end in local_ranges.values())
    mapping = np.full(local_norb, -1, dtype=np.int64)
    for atom, (local_start, local_end) in local_ranges.items():
        global_start, global_end = global_ranges[atom]
        width = local_end - local_start
        if global_end - global_start != width:
            raise ValueError(f"Orbital count mismatch for atom {atom}")
        mapping[local_start:local_end] = np.arange(global_start, global_end, dtype=np.int64)
    if np.any(mapping < 0):
        # Core matrices should already be filtered to core orbitals.  A negative
        # entry would indicate that filtering failed or non-core rows survived.
        pass
    return mapping


def read_density_matrix_metadata(path: str | os.PathLike[str]) -> SiestaMatrixMetadata:
    """Read header metadata from a SIESTA `.DM` file."""

    path = Path(path)
    with _FortranRecordReader(path) as reader:
        header = reader.read_record()
    values = _unpack_ints(header)
    if len(values) < 2:
        raise ValueError(f"Invalid SIESTA DM header in {path}")
    nsc = tuple(values[2:5]) if len(values) >= 5 else None
    return SiestaMatrixMetadata(
        kind="DM",
        path=path,
        norbitals=values[0],
        nspin=values[1],
        nsc=nsc,
    )


def read_density_matrix_sparse(path: str | os.PathLike[str]) -> SiestaDensityMatrix:
    """Read sparse density matrix arrays from a SIESTA `.DM` file."""

    path = Path(path)
    with _FortranRecordReader(path) as reader:
        header = _unpack_ints(reader.read_record())
        if len(header) < 2:
            raise ValueError(f"Invalid SIESTA DM header in {path}")
        norbitals = header[0]
        nspin = header[1]
        nsc = tuple(header[2:5]) if len(header) >= 5 else None
        rows, cols, numh = _read_sparse_pattern(reader, norbitals)
        density = _read_sparse_value_rows(reader, numh, nspin, double_precision=True)

    return SiestaDensityMatrix(
        metadata=SiestaMatrixMetadata(
            kind="DM",
            path=path,
            norbitals=norbitals,
            nspin=nspin,
            nsc=nsc,
        ),
        rows=rows,
        cols=cols,
        density=density,
    )


def read_hsx_metadata(path: str | os.PathLike[str]) -> SiestaMatrixMetadata:
    """Read header metadata from a SIESTA `.HSX` file.

    Supports the modern version-1 format written by `Src/io_hsx.F90`; a minimal
    fallback is kept for the older version-0 layout used by utility code.
    """

    path = Path(path)
    with _FortranRecordReader(path) as reader:
        first = _unpack_ints(reader.read_record())
        if first == [1]:
            double_precision = _unpack_logical(reader.read_record())
            values = _unpack_ints(reader.read_record())
            if len(values) < 7:
                raise ValueError(f"Invalid SIESTA HSX v1 header in {path}")
            return SiestaMatrixMetadata(
                kind="HSX",
                path=path,
                natoms=values[0],
                norbitals=values[1],
                nspin=values[2],
                nspecies=values[3],
                nsc=tuple(values[4:7]),
                version=1,
                double_precision=double_precision,
            )
        if len(first) >= 4:
            return SiestaMatrixMetadata(
                kind="HSX",
                path=path,
                norbitals=first[0],
                nspin=first[2],
                version=0,
            )
    raise ValueError(f"Invalid SIESTA HSX header in {path}")


def read_hsx_sparse(path: str | os.PathLike[str]) -> SiestaHsxMatrix:
    """Read sparse Hamiltonian and overlap arrays from a SIESTA `.HSX` file.

    The returned row/column indices are zero-based.  Hamiltonian values are
    shaped `(nspin, nnz)`, while overlap values are shaped `(nnz,)`.
    """

    path = Path(path)
    with _FortranRecordReader(path) as reader:
        first = _unpack_ints(reader.read_record())
        if first != [1]:
            raise ValueError(f"Only SIESTA HSX version 1 is supported for sparse value reads: {path}")
        double_precision = _unpack_logical(reader.read_record())
        values = _unpack_ints(reader.read_record())
        if len(values) < 7:
            raise ValueError(f"Invalid SIESTA HSX v1 header in {path}")
        natoms, norbitals, nspin, nspecies, *nsc = values[:7]
        metadata = SiestaMatrixMetadata(
            kind="HSX",
            path=path,
            natoms=natoms,
            norbitals=norbitals,
            nspin=nspin,
            nspecies=nspecies,
            nsc=tuple(nsc),
            version=1,
            double_precision=double_precision,
        )

        reader.read_record()  # ucell, Ef, qtot, temp
        reader.read_record()  # isc_off, xa, isa, lasto
        species_payload = reader.read_record()
        species_orbital_counts = _unpack_species_orbital_counts(species_payload, nspecies)
        for _ in species_orbital_counts:
            reader.read_record()

        rows, cols, numh = _read_sparse_pattern(reader, norbitals)
        hamiltonian = _read_sparse_value_rows(reader, numh, nspin, double_precision)

        overlap = np.empty(rows.size, dtype=np.float64)
        cursor = 0
        for row_nnz in numh:
            row_values = _unpack_reals(reader.read_record(), double_precision)
            if row_values.size != row_nnz:
                raise ValueError(f"HSX overlap row value count mismatch in {path}")
            overlap[cursor : cursor + row_nnz] = row_values
            cursor += row_nnz

    return SiestaHsxMatrix(
        metadata=metadata,
        rows=rows,
        cols=cols,
        hamiltonian=hamiltonian,
        overlap=overlap,
    )


class _FortranRecordReader:
    """Reader for little-endian gfortran sequential unformatted records."""

    def __init__(self, path: Path):
        self.path = path
        self.handle = None

    def __enter__(self):
        self.handle = self.path.open("rb")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.handle.close()

    def read_record(self) -> bytes:
        marker = self.handle.read(4)
        if len(marker) != 4:
            raise EOFError(f"Unexpected end of Fortran record in {self.path}")
        (length,) = struct.unpack("<i", marker)
        if length < 0:
            raise ValueError(f"Unsupported split Fortran record in {self.path}")
        payload = self.handle.read(length)
        trailer = self.handle.read(4)
        if len(payload) != length or len(trailer) != 4:
            raise EOFError(f"Truncated Fortran record in {self.path}")
        (trailer_length,) = struct.unpack("<i", trailer)
        if trailer_length != length:
            raise ValueError(f"Fortran record marker mismatch in {self.path}")
        return payload


def _read_matrix_metadata(dm_path: Path | None, hsx_path: Path | None) -> dict[str, dict]:
    metadata = {}
    if dm_path is not None:
        metadata["density"] = read_density_matrix_metadata(dm_path).to_metadata()
    if hsx_path is not None:
        metadata["hamiltonian_overlap"] = read_hsx_metadata(hsx_path).to_metadata()
    return metadata


def _read_elsi_log_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    complete = True
    parse_error = None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        complete = False
        parse_error = str(exc)
        payload = _read_partial_elsi_log_records(path.read_text())
    if not isinstance(payload, list):
        return {}
    solver_chosen = sorted({entry.get("solver_chosen") for entry in payload if entry.get("solver_chosen")})
    solver_used = sorted({entry.get("solver_used") for entry in payload if entry.get("solver_used")})
    last_settings = {}
    if payload:
        last_settings = payload[-1].get("solver_settings", {}) or {}
    metadata = {
        "complete": complete,
        "num_records": len(payload),
        "solver_chosen": solver_chosen,
        "solver_used": solver_used,
        "last_solver_settings": last_settings,
    }
    if parse_error is not None:
        metadata["parse_error"] = parse_error
    return metadata


def _read_partial_elsi_log_records(text: str) -> list[dict]:
    decoder = json.JSONDecoder()
    records = []
    cursor = text.find("[")
    if cursor < 0:
        return records
    cursor += 1
    while cursor < len(text):
        while cursor < len(text) and text[cursor] in " \t\r\n,":
            cursor += 1
        if cursor >= len(text) or text[cursor] == "]":
            break
        try:
            item, cursor = decoder.raw_decode(text, cursor)
        except json.JSONDecodeError:
            break
        if isinstance(item, dict):
            records.append(item)
    return records


def _read_core_matrix_metadata(
    hsx_path: Path | None,
    dm_path: Path | None,
    core_atom_orbital_ranges: dict[int, tuple[int, int]],
) -> dict[str, dict]:
    if not core_atom_orbital_ranges:
        return {}
    metadata = {}
    if dm_path is not None:
        dm = read_density_matrix_sparse(dm_path)
        metadata["density"] = dm.core_block(core_atom_orbital_ranges).to_metadata()
    if hsx_path is not None:
        hsx = read_hsx_sparse(hsx_path)
        metadata["hamiltonian_overlap"] = hsx.core_block(core_atom_orbital_ranges).to_metadata()
    return metadata


def _validate_matrix_orbital_count(block_id: int, result: dict) -> None:
    atom_ranges = result.get("atom_orbital_ranges", {})
    if not atom_ranges:
        return
    expected_norb = max(int(bounds[1]) for bounds in atom_ranges.values())
    matrix_metadata = result.get("matrix_metadata", {})
    for name, metadata in matrix_metadata.items():
        norb = metadata.get("norbitals")
        if norb is not None and int(norb) != expected_norb:
            raise ValueError(
                f"SIESTA block {block_id} {name} orbital count {norb} "
                f"does not match ORB_INDX count {expected_norb}"
            )


def _unpack_ints(payload: bytes) -> list[int]:
    if len(payload) % 4 != 0:
        raise ValueError("Integer record payload length is not divisible by 4")
    return list(struct.unpack("<" + "i" * (len(payload) // 4), payload))


def _read_sparse_pattern(reader: "_FortranRecordReader", norbitals: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    numh = np.asarray(_unpack_ints(reader.read_record()), dtype=np.int64)
    if numh.size != norbitals:
        raise ValueError(f"Sparse row count length {numh.size} does not match norbitals {norbitals} in {reader.path}")
    rows = np.repeat(np.arange(norbitals, dtype=np.int64), numh)
    cols = []
    for row_nnz in numh:
        row_cols = np.asarray(_unpack_ints(reader.read_record()), dtype=np.int64)
        if row_cols.size != row_nnz:
            raise ValueError(f"Sparse row column count mismatch in {reader.path}")
        cols.append(row_cols - 1)
    cols = np.concatenate(cols) if cols else np.asarray([], dtype=np.int64)
    return rows, cols, numh


def _read_sparse_value_rows(
    reader: "_FortranRecordReader",
    numh: np.ndarray,
    nspin: int,
    double_precision: bool,
) -> np.ndarray:
    values = np.empty((nspin, int(numh.sum())), dtype=np.float64)
    for spin in range(nspin):
        cursor = 0
        for row_nnz in numh:
            row_values = _unpack_reals(reader.read_record(), double_precision)
            if row_values.size != row_nnz:
                raise ValueError(f"Sparse row value count mismatch in {reader.path}")
            values[spin, cursor : cursor + row_nnz] = row_values
            cursor += row_nnz
    return values


def _unpack_reals(payload: bytes, double_precision: bool) -> np.ndarray:
    dtype = "<f8" if double_precision else "<f4"
    itemsize = 8 if double_precision else 4
    if len(payload) % itemsize != 0:
        raise ValueError("Real record payload length is invalid")
    return np.frombuffer(payload, dtype=dtype).astype(np.float64)


def _unpack_species_orbital_counts(payload: bytes, nspecies: int) -> list[int]:
    # SIESTA writes (label, zval, no) where label is char(len=20), zval is
    # double precision, and no is a 32-bit integer.
    stride = 32
    if len(payload) != nspecies * stride:
        raise ValueError("Unexpected HSX species record length")
    counts = []
    for offset in range(0, len(payload), stride):
        counts.append(struct.unpack("<i", payload[offset + 28 : offset + 32])[0])
    return counts


def _unpack_logical(payload: bytes) -> bool:
    values = _unpack_ints(payload)
    if len(values) != 1:
        raise ValueError("Logical record has unexpected length")
    return values[0] != 0


def _read_json_list(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"{path} does not contain a JSON list")
    return payload


def _read_optional_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _option_key(lowered_line: str) -> str:
    if not lowered_line or lowered_line.startswith("#"):
        return ""
    return lowered_line.split()[0].replace(".", "")


def _render_run_script(siesta_bin: str | None, threads_per_proc: int) -> str:
    siesta_bin = siesta_bin or "${EWF_SIESTA_BIN:-siesta}"
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"export OMP_NUM_THREADS=${{EWF_THREADS_PER_PROC:-{threads_per_proc}}}",
            f"export MKL_NUM_THREADS=${{EWF_THREADS_PER_PROC:-{threads_per_proc}}}",
            f"export OPENBLAS_NUM_THREADS=${{EWF_THREADS_PER_PROC:-{threads_per_proc}}}",
            f'"{siesta_bin}" input.fdf > siesta.out 2>&1',
            "",
        ]
    )


def _get_int(environ: dict[str, str], name: str, default: int) -> int:
    value = int(environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _get_optional_int(environ: dict[str, str], name: str) -> int | None:
    if name not in environ or environ[name] == "":
        return None
    value = int(environ[name])
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _get_nonnegative_int(environ: dict[str, str], name: str, default: int) -> int:
    value = int(environ.get(name, default))
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _get_positive_float(environ: dict[str, str], name: str, default: float) -> float:
    value = float(environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _get_nonnegative_float(environ: dict[str, str], name: str, default: float) -> float:
    value = float(environ.get(name, default))
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _get_bool(environ: dict[str, str], name: str, default: bool) -> bool:
    raw = environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _detect_convergence(text: str) -> bool | None:
    lower = text.lower()
    if "scf converged" in lower or "scf cycle converged" in lower:
        return True
    if "not converged" in lower or "scf did not converge" in lower:
        return False
    return None


def _detect_total_energy_ev(text: str) -> float | None:
    patterns = (
        r"siesta:\s*E_KS\(eV\)\s*=\s*([-+0-9.Ee]+)",
        r"siesta:\s*Total\s*=\s*([-+0-9.Ee]+)",
        r"Total\s+energy\s*[:=]\s*([-+0-9.Ee]+)\s*eV",
        r"^\s*scf:\s+\d+\s+[-+0-9.Ee]+\s+([-+0-9.Ee]+)\s+[-+0-9.Ee]+",
    )
    for pattern in patterns:
        flags = re.MULTILINE if pattern.startswith("^") else 0
        matches = re.findall(pattern, text, flags=flags)
        if matches:
            return float(matches[-1])
    return None


def _detect_wall_time_seconds(text: str) -> float | None:
    patterns = (
        r"timer:\s*Elapsed\s+wall\s+time\s*\(sec\)\s*=\s*([-+0-9.Ee]+)",
        r"Elapsed\s+wall\s+time\s*[:=]\s*([-+0-9.Ee]+)\s*s",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return float(matches[-1])
    return None


def _detect_run_diagnostics(text: str) -> dict[str, object]:
    scf_steps = []
    for match in re.finditer(r"^\s*scf:\s+(\d+)\s+(.+)$", text, flags=re.MULTILINE | re.IGNORECASE):
        values = _parse_float_fields(match.group(2))
        scf_steps.append(
            {
                "step": int(match.group(1)),
                "values": values,
                "total_energy_ev": values[1] if len(values) > 1 else None,
            }
        )

    diagnostics: dict[str, object] = {
        "num_scf_steps": len(scf_steps),
        "last_scf_step": scf_steps[-1]["step"] if scf_steps else None,
        "last_scf_energy_ev": scf_steps[-1]["total_energy_ev"] if scf_steps else None,
        "convergence_reason": _detect_convergence_reason(text),
    }
    if scf_steps:
        diagnostics["first_scf_energy_ev"] = scf_steps[0]["total_energy_ev"]
    return diagnostics


def _parse_float_fields(text: str) -> list[float]:
    values = []
    for field in text.split():
        try:
            values.append(float(field))
        except ValueError:
            pass
    return values


def _detect_convergence_reason(text: str) -> str | None:
    lower = text.lower()
    if "scf_not_conv" in lower or "scf did not converge" in lower:
        return "max_scf_iterations_or_required_convergence_not_met"
    if "scf converged" in lower or "scf cycle converged" in lower:
        return "scf_converged"
    if "elpa" in lower and "ntpoly" not in lower:
        return "non_ntpoly_solver_output_detected"
    return None


_DENSITY_MATRIX_CANDIDATES = ("DM", "DMHS", "block.DM", "siesta.DM", "*.DM")
_HAMILTONIAN_MATRIX_CANDIDATES = ("HSX", "H", "block.HSX", "siesta.HSX", "*.HSX")
_OVERLAP_MATRIX_CANDIDATES = ("HSX", "S", "block.HSX", "siesta.HSX", "*.HSX")
_ORBITAL_INDEX_CANDIDATES = ("ORB_INDX", "block.ORB_INDX", "siesta.ORB_INDX", "*.ORB_INDX")


def _first_existing(block_dir: Path, names: Sequence[str], block_id: int) -> Path | None:
    label = f"block_{block_id:04d}"
    label_names = [name.replace("block", label, 1) for name in names if name.startswith("block.")]
    for name in label_names:
        path = block_dir / name
        if path.exists():
            return path
    for name in names:
        if "*" in name:
            matches = sorted(block_dir.glob(name))
            if matches:
                return matches[0]
            continue
        path = block_dir / name
        if path.exists():
            return path
    return None


def _read_atom_orbital_ranges(
    orbital_index_path: Path | None,
    local_to_global_atom_index: Sequence[int],
) -> dict[int, tuple[int, int]]:
    if orbital_index_path is None:
        return {}

    local_ranges: dict[int, list[int]] = {}
    for line in orbital_index_path.read_text(errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        try:
            orbital = int(fields[0]) - 1
            local_atom = int(fields[1]) - 1
        except ValueError:
            continue
        if orbital < 0 or local_atom < 0:
            continue
        local_ranges.setdefault(local_atom, [orbital, orbital + 1])
        local_ranges[local_atom][1] = orbital + 1

    ranges = {}
    for local_atom, (start, end) in local_ranges.items():
        if local_atom < len(local_to_global_atom_index):
            ranges[int(local_to_global_atom_index[local_atom])] = (start, end)
    return ranges


def _path_to_str(path: Path | None) -> str | None:
    return None if path is None else str(path)


def _str_to_path(path: str | None) -> Path | None:
    return None if path is None else Path(path)
