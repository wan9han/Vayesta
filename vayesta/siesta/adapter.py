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
            "validation": None,
            "ewf_results": None,
            "global_matrices": None,
            "electron_constraint": None,
            "physical_readiness": None,
        }
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
            payload["electron_constraint"] = write_electron_constraint_manifest(
                self.config.workdir,
                self.fdf,
            )
            payload["validation"] = write_validation_manifest(
                self.config.workdir,
                natoms=self.natoms,
                min_buffer_atoms=minimum_buffer_atoms(self.config),
            )
        if (self.config.workdir / "validation.json").exists():
            payload["physical_readiness"] = write_physical_readiness_manifest(self.config.workdir)
        return payload


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

    bonds = []
    atoms = fdf.atoms
    for i, atom_i in enumerate(atoms):
        radius_i = _covalent_radius(fdf, atom_i.species)
        for atom_j in atoms[i + 1 :]:
            radius_j = _covalent_radius(fdf, atom_j.species)
            distance = _atom_distance(atom_i, atom_j)
            cutoff = radius_i + radius_j + tolerance_angstrom
            if distance <= cutoff:
                bonds.append(
                    {
                        "atom_i": atom_i.global_index,
                        "atom_j": atom_j.global_index,
                        "species_i": fdf.species_labels.get(atom_i.species, str(atom_i.species)),
                        "species_j": fdf.species_labels.get(atom_j.species, str(atom_j.species)),
                        "distance_angstrom": round(distance, 6),
                        "cutoff_angstrom": round(cutoff, 6),
                    }
                )
    return bonds


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
        "electron_policy": "diagnostic_density_overlap_trace_only",
        "energy_policy": "diagnostic_block_sum_no_double_counting_correction",
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


def build_boundary_correction_plan(embedding_contract: dict) -> dict:
    """Build placeholder correction slots for each pending boundary embedding term."""

    corrections = []
    for term in embedding_contract.get("terms", []):
        if term.get("status") != "pending_embedding_correction":
            continue
        corrections.append(
            {
                "block_id": term["block_id"],
                "bond_atoms": term["bond_atoms"],
                "core_atom": term["core_atom"],
                "environment_atom": term["environment_atom"],
                "correction_type": "boundary_bond_embedding",
                "hamiltonian_embedding_potential": None,
                "energy_correction_ev": None,
                "electron_count_correction": None,
                "status": "not_parameterized",
            }
        )
    return {
        "version": 1,
        "correction_level": "placeholder",
        "num_corrections": len(corrections),
        "num_unparameterized_corrections": len(corrections),
        "corrections": corrections,
    }


def write_boundary_corrections_manifest(workdir: str | os.PathLike[str]) -> dict:
    """Write `boundary_corrections.json` from `embedding_contract.json`."""

    workdir = Path(workdir)
    contract_path = workdir / "embedding_contract.json"
    if not contract_path.exists():
        raise FileNotFoundError(contract_path)
    payload = build_boundary_correction_plan(json.loads(contract_path.read_text()))
    (workdir / "boundary_corrections.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
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
    return {
        "num_machines": num_machines,
        "procs_per_machine": procs_per_machine,
        "num_ranks": num_machines * procs_per_machine,
        "num_blocks": len(blocks),
        "ranks": ranks,
        "block_owner_rank": block_owner,
    }


def write_schedule_manifest(
    workdir: str | os.PathLike[str],
    blocks: Sequence[SiestaBlock],
    num_machines: int,
    procs_per_machine: int,
) -> dict:
    """Write `schedule.json` with the complete planned block distribution."""

    workdir = Path(workdir)
    payload = build_schedule(blocks, num_machines, procs_per_machine)
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
        fragment.siesta_total_energy_ev = result.total_energy_ev
        fragment.siesta_density_matrix_path = result.density_matrix_path
        fragment.siesta_hamiltonian_matrix_path = result.hamiltonian_matrix_path
        fragment.siesta_overlap_matrix_path = result.overlap_matrix_path
        fragment.siesta_orbital_index_path = result.orbital_index_path
        fragment.siesta_core_atom_orbital_ranges = dict(result.core_atom_orbital_ranges)
        fragment.siesta_core_matrix_metadata = dict(result.core_matrix_metadata)
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
    return attach_siesta_results_to_fragments(fragments, results, strict=strict)


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
    return {
        "workdir": str(workdir),
        "num_blocks": len(blocks),
        "num_results": len(results),
        "num_successful_results": sum(1 for code in returncodes if code == 0),
        "num_failed_results": sum(1 for code in returncodes if code != 0),
        "num_converged_results": sum(1 for item in converged if item),
        "num_unconverged_results": sum(1 for item in converged if not item),
        "num_scheduled_ranks": len(scheduled_ranks),
        "scheduled_ranks": scheduled_ranks,
        "num_ranks_with_results": len(ranks),
        "ranks_with_results": ranks,
        "num_machines": len(machines),
        "machines": machines,
        "total_wall_time_seconds": None if not wall_times else float(sum(wall_times)),
        "max_block_wall_time_seconds": None if not wall_times else float(max(wall_times)),
        "min_block_wall_time_seconds": None if not wall_times else float(min(wall_times)),
        "mean_block_wall_time_seconds": None if not wall_times else float(sum(wall_times) / len(wall_times)),
        "solver_used": _summarize_solver_used(results),
        "ntpoly_methods": _summarize_ntpoly_methods(results),
        "max_scf_steps": _summarize_max_scf_steps(results),
        "blocks": block_summaries,
    }


def write_run_summary_manifest(workdir: str | os.PathLike[str]) -> dict:
    """Write `run_summary.json` with scheduling, success, and timing metrics."""

    workdir = Path(workdir)
    payload = summarize_run(workdir)
    (workdir / "run_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def compare_weak_scaling_runs(workdirs: Sequence[str | os.PathLike[str]]) -> dict:
    """Compare multiple `run_summary.json` files for weak-scaling analysis."""

    if not workdirs:
        raise ValueError("At least one workdir is required")
    summaries = [_load_or_create_run_summary(Path(workdir)) for workdir in workdirs]
    baseline_time = _first_nonnull(summary.get("max_block_wall_time_seconds") for summary in summaries)
    runs = [_weak_scaling_run_entry(summary, baseline_time) for summary in summaries]
    return {
        "baseline_workdir": runs[0]["workdir"],
        "baseline_max_block_wall_time_seconds": baseline_time,
        "num_runs": len(runs),
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


def build_electron_constraint(fdf: FdfInput, global_matrices_metadata: dict) -> dict:
    """Build a diagnostic electron-number constraint from valence counts and Tr(D S)."""

    target = estimate_valence_electron_count(fdf)
    observed = global_matrices_metadata.get("density_overlap_trace_total")
    deviation = None if observed is None else float(observed - target)
    return {
        "constraint_level": "diagnostic",
        "target_valence_electrons": float(target),
        "observed_density_overlap_trace": observed,
        "electron_count_deviation": deviation,
        "chemical_potential_status": "not_applied",
        "policy": "report_only_no_mu_update",
    }


def write_electron_constraint_manifest(
    workdir: str | os.PathLike[str],
    fdf: FdfInput,
) -> dict:
    """Write `electron_constraint.json` using `global_matrices.json`."""

    workdir = Path(workdir)
    global_path = workdir / "global_matrices.json"
    if not global_path.exists():
        raise FileNotFoundError(global_path)
    payload = build_electron_constraint(fdf, json.loads(global_path.read_text()))
    (workdir / "electron_constraint.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def build_physical_readiness_report(workdir: str | os.PathLike[str]) -> dict:
    """Report whether SIESTA block artifacts are ready for physical EWF use."""

    workdir = Path(workdir)
    validation = _read_optional_json(workdir / "validation.json") or {}
    embedding = _read_optional_json(workdir / "embedding_contract.json") or {}
    corrections = _read_optional_json(workdir / "boundary_corrections.json") or {}
    electron = _read_optional_json(workdir / "electron_constraint.json") or {}
    global_matrices = _read_optional_json(workdir / "global_matrices.json") or {}

    backend_ready = bool(validation.get("ok"))
    blockers = []
    if not backend_ready:
        blockers.append("SIESTA backend artifacts failed validation or validation.json is missing")

    pending_embedding = int(embedding.get("num_pending_embedding_terms", 0))
    if pending_embedding:
        blockers.append(f"{pending_embedding} boundary embedding terms do not have embedding potentials")

    unparameterized = int(corrections.get("num_unparameterized_corrections", 0))
    if unparameterized:
        blockers.append(f"{unparameterized} boundary correction slots are not parameterized")

    if backend_ready and not electron:
        blockers.append("electron_constraint.json is missing")
    elif electron and electron.get("chemical_potential_status") != "applied":
        blockers.append("global electron-number or chemical-potential constraint is not applied")

    embedded_ready = backend_ready and not blockers
    return {
        "version": 1,
        "backend_artifacts_ready": backend_ready,
        "embedded_observable_ready": embedded_ready,
        "status": "embedded_observable_ready" if embedded_ready else "diagnostic_backend_only",
        "blockers": blockers,
        "diagnostic_outputs": {
            "energy_policy": validation.get("energy_policy"),
            "density_overlap_trace_total": global_matrices.get("density_overlap_trace_total"),
            "electron_count_deviation": electron.get("electron_count_deviation"),
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
        nt_method = elsi_metadata.get("last_solver_settings", {}).get("nt_method")
        if nt_method is not None and int(nt_method) != 2:
            errors.append(f"Block {result.block_id} used NTPoly method {nt_method}, expected TRS2 method 2")
        if min_buffer_atoms > 0 and _is_internal_block(result, natoms) and len(result.buffer_atoms) < min_buffer_atoms:
            warnings.append(
                f"Block {result.block_id} has {len(result.buffer_atoms)} buffer atoms; "
                f"requested at least {min_buffer_atoms}"
            )

    total_energy = _sum_block_energies(results)
    if total_energy is not None:
        warnings.append(
            "total_block_energy_ev is a diagnostic sum of independent block energies; "
            "it is not an embedded total energy."
        )
    warnings.append(
        "Current matrices are core-owned SIESTA block collections without boundary embedding potential, "
        "chemical-potential constraint, or double-counting energy correction."
    )
    if global_matrices is not None:
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
        energy_policy="diagnostic_block_sum_not_embedded_total" if total_energy is not None else "not_available",
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
    errors = [
        f"Block {term['block_id']} embedding term {term['bond_atoms']} has uncovered boundary"
        for term in payload.get("terms", [])
        if term.get("status") == "invalid_uncovered_boundary"
    ]
    pending = int(payload.get("num_pending_embedding_terms", 0))
    warnings = []
    if pending:
        warnings.append(
            f"{pending} boundary embedding terms require embedding potential and energy correction; "
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


def _load_or_create_run_summary(workdir: Path) -> dict:
    summary_path = workdir / "run_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    return summarize_run(workdir)


def _weak_scaling_run_entry(summary: dict, baseline_time: float | None) -> dict:
    max_time = summary.get("max_block_wall_time_seconds")
    success_rate = _ratio(summary.get("num_successful_results"), summary.get("num_blocks"))
    converged_rate = _ratio(summary.get("num_converged_results"), summary.get("num_blocks"))
    return {
        "workdir": summary.get("workdir"),
        "num_blocks": summary.get("num_blocks"),
        "num_scheduled_ranks": summary.get("num_scheduled_ranks"),
        "num_ranks_with_results": summary.get("num_ranks_with_results"),
        "num_machines": summary.get("num_machines"),
        "num_successful_results": summary.get("num_successful_results"),
        "num_failed_results": summary.get("num_failed_results"),
        "success_rate": success_rate,
        "converged_rate": converged_rate,
        "max_block_wall_time_seconds": max_time,
        "mean_block_wall_time_seconds": summary.get("mean_block_wall_time_seconds"),
        "solver_used": summary.get("solver_used", []),
        "ntpoly_methods": summary.get("ntpoly_methods", []),
        "max_scf_steps": summary.get("max_scf_steps"),
        "weak_scaling_efficiency_vs_baseline": _weak_scaling_efficiency(baseline_time, max_time),
    }


def _weak_scaling_efficiency(baseline_time: float | None, current_time: float | None) -> float | None:
    if baseline_time is None or current_time is None or current_time <= 0:
        return None
    return float(baseline_time / current_time)


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
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        return {}
    solver_chosen = sorted({entry.get("solver_chosen") for entry in payload if entry.get("solver_chosen")})
    solver_used = sorted({entry.get("solver_used") for entry in payload if entry.get("solver_used")})
    last_settings = {}
    if payload:
        last_settings = payload[-1].get("solver_settings", {}) or {}
    return {
        "num_records": len(payload),
        "solver_chosen": solver_chosen,
        "solver_used": solver_used,
        "last_solver_settings": last_settings,
    }


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
