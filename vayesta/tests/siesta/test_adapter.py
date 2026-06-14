import json
import os
import subprocess
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

from vayesta.siesta import adapter


def _write_fortran_record(handle, payload):
    handle.write(struct.pack("<i", len(payload)))
    handle.write(payload)
    handle.write(struct.pack("<i", len(payload)))


def _write_minimal_dm(path, norbitals=3, nspin=1, nsc=(1, 1, 1)):
    with path.open("wb") as handle:
        _write_fortran_record(handle, struct.pack("<5i", norbitals, nspin, *nsc))


def _write_full_dm(path, numh, columns, density_rows, nsc=(1, 1, 1)):
    with path.open("wb") as handle:
        norbitals = len(numh)
        nspin = len(density_rows)
        _write_fortran_record(handle, struct.pack("<5i", norbitals, nspin, *nsc))
        _write_fortran_record(handle, struct.pack("<" + "i" * norbitals, *numh))
        for row_cols in columns:
            _write_fortran_record(handle, struct.pack("<" + "i" * len(row_cols), *row_cols))
        for spin_rows in density_rows:
            for row_values in spin_rows:
                _write_fortran_record(handle, struct.pack("<" + "d" * len(row_values), *row_values))


def _write_minimal_hsx(path, natoms=2, norbitals=3, nspin=1, nspecies=2, nsc=(1, 1, 1)):
    with path.open("wb") as handle:
        _write_fortran_record(handle, struct.pack("<i", 1))
        _write_fortran_record(handle, struct.pack("<i", 1))
        _write_fortran_record(handle, struct.pack("<7i", natoms, norbitals, nspin, nspecies, *nsc))


def _write_full_hsx(path, numh, columns, hamiltonian_rows, overlap_rows, natoms=1, nspecies=1):
    with path.open("wb") as handle:
        norbitals = len(numh)
        nspin = len(hamiltonian_rows)
        _write_fortran_record(handle, struct.pack("<i", 1))
        _write_fortran_record(handle, struct.pack("<i", 1))
        _write_fortran_record(handle, struct.pack("<7i", natoms, norbitals, nspin, nspecies, 1, 1, 1))
        _write_fortran_record(handle, struct.pack("<12d", *([0.0] * 12)))
        _write_fortran_record(handle, struct.pack("<4i", 0, 0, 0, 0))
        label = b"H" + b" " * 19
        _write_fortran_record(handle, label + struct.pack("<di", 1.0, 1))
        _write_fortran_record(handle, struct.pack("<3i", 1, 0, 1))
        _write_fortran_record(handle, struct.pack("<" + "i" * norbitals, *numh))
        for row_cols in columns:
            _write_fortran_record(handle, struct.pack("<" + "i" * len(row_cols), *row_cols))
        for spin_rows in hamiltonian_rows:
            for row_values in spin_rows:
                _write_fortran_record(handle, struct.pack("<" + "d" * len(row_values), *row_values))
        for row_values in overlap_rows:
            _write_fortran_record(handle, struct.pack("<" + "d" * len(row_values), *row_values))


def test_partition_contiguous_atoms_with_buffer():
    blocks = adapter.partition_contiguous_atoms(10, block_atoms=4, buffer_atoms=1, num_machines=2)

    assert [(b.core_atom_start, b.core_atom_end) for b in blocks] == [(0, 4), (4, 8), (8, 10)]
    assert [(b.input_atom_start, b.input_atom_end) for b in blocks] == [(0, 5), (3, 9), (7, 10)]
    assert [b.machine_id for b in blocks] == [0, 1, 0]


def test_partition_contiguous_atom_groups_aligns_core_and_buffer_to_groups():
    blocks = adapter.partition_contiguous_atom_groups(
        20,
        group_size_atoms=6,
        block_groups=1,
        buffer_groups=1,
        num_machines=2,
    )

    assert [(b.core_atom_start, b.core_atom_end) for b in blocks] == [(0, 6), (6, 12), (12, 20)]
    assert [(b.input_atom_start, b.input_atom_end) for b in blocks] == [(0, 12), (0, 20), (6, 20)]
    assert [b.machine_id for b in blocks] == [0, 1, 0]


def test_partition_contiguous_atom_groups_can_keep_terminal_caps_with_end_groups():
    blocks = adapter.partition_contiguous_atom_groups(
        26,
        group_size_atoms=6,
        block_groups=1,
        buffer_groups=1,
        terminal_cap_atoms=2,
        num_machines=4,
    )

    assert [(b.core_atom_start, b.core_atom_end) for b in blocks] == [(0, 7), (7, 13), (13, 19), (19, 26)]
    assert [(b.input_atom_start, b.input_atom_end) for b in blocks] == [(0, 13), (0, 19), (7, 26), (13, 26)]
    assert [b.machine_id for b in blocks] == [0, 1, 2, 3]


def test_assign_blocks_to_rank_within_machine():
    blocks = adapter.partition_contiguous_atoms(24, block_atoms=2, buffer_atoms=0, num_machines=2)

    assert [b.block_id for b in adapter.assign_blocks_to_rank(blocks, 0, 0, 3)] == [0, 6]
    assert [b.block_id for b in adapter.assign_blocks_to_rank(blocks, 0, 1, 3)] == [2, 8]
    assert [b.block_id for b in adapter.assign_blocks_to_rank(blocks, 0, 2, 3)] == [4, 10]
    assert [b.block_id for b in adapter.assign_blocks_to_rank(blocks, 1, 0, 3)] == [1, 7]


def test_write_schedule_manifest_covers_all_ranks_and_blocks(tmp_path):
    blocks = adapter.partition_contiguous_atoms(24, block_atoms=4, buffer_atoms=1, num_machines=2)

    payload = adapter.write_schedule_manifest(tmp_path, blocks, num_machines=2, procs_per_machine=2)

    assert payload["num_ranks"] == 4
    assert payload["total_ranks"] == 4
    assert payload["threads_per_proc"] is None
    assert payload["num_blocks"] == 6
    assert payload["rank_block_counts"] == {"0": 2, "1": 1, "2": 2, "3": 1}
    assert payload["ranks"][0]["rank"] == 0
    assert payload["ranks"][0]["block_ids"] == [0, 4]
    assert payload["ranks"][1]["block_ids"] == [2]
    assert payload["ranks"][2]["block_ids"] == [1, 5]
    assert payload["ranks"][3]["block_ids"] == [3]
    assert payload["block_owner_rank"] == {"0": 0, "1": 2, "2": 1, "3": 3, "4": 0, "5": 2}
    assert payload["ranks"][0]["blocks"][0]["core_atoms"] == 4
    assert payload["ranks"][0]["blocks"][0]["buffer_atoms"] == 1
    assert json.loads((tmp_path / "schedule.json").read_text()) == payload


def test_add_siesta_block_fragments_uses_core_atoms_and_keeps_buffer_metadata():
    class FakeFragmentation:
        def __init__(self):
            self.calls = []

        def add_atomic_fragment(self, atoms, orbital_filter=None, name=None):
            fragment = type("FakeFragment", (), {})()
            fragment.atoms = list(atoms)
            fragment.name = name
            fragment.orbital_filter = orbital_filter
            self.calls.append((list(atoms), orbital_filter, name, fragment))
            return fragment

    blocks = adapter.partition_contiguous_atoms(10, block_atoms=4, buffer_atoms=1, num_machines=2)
    fragmentation = FakeFragmentation()

    fragments = adapter.add_siesta_block_fragments(fragmentation, blocks, orbital_filter="2p")

    assert len(fragments) == 3
    assert [call[0] for call in fragmentation.calls] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert [fragment.name for fragment in fragments] == ["siesta-block-0000", "siesta-block-0001", "siesta-block-0002"]
    assert fragments[1].orbital_filter == "2p"
    assert fragments[1].siesta_block_id == 1
    assert fragments[1].siesta_machine_id == 1
    assert fragments[1].siesta_core_atoms == (4, 5, 6, 7)
    assert fragments[1].siesta_input_atoms == (3, 4, 5, 6, 7, 8)
    assert fragments[1].siesta_buffer_atoms == (3, 8)
    assert fragments[1].siesta_core_atom_range == (4, 8)
    assert fragments[1].siesta_input_atom_range == (3, 9)


def test_add_siesta_block_fragments_registers_real_vayesta_fragments():
    import pyscf.gto
    import pyscf.scf
    import vayesta.ewf

    mol = pyscf.gto.Mole()
    mol.atom = "H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22"
    mol.basis = "sto-6g"
    mol.verbose = 0
    mol.build()
    mf = pyscf.scf.RHF(mol).run(verbose=0)
    emb = vayesta.ewf.EWF(mf, solver="MP2", bath_options={"threshold": -1})
    blocks = adapter.partition_contiguous_atoms(mol.natm, block_atoms=2, buffer_atoms=1, num_machines=2)

    with emb.sao_fragmentation() as frag:
        created = adapter.add_siesta_block_fragments(frag, blocks)

    assert len(created) == 2
    assert len(emb.fragments) == 2
    assert [fragment.atoms for fragment in emb.fragments] == [[0, 1], [2, 3]]
    assert emb.fragments[0].siesta_core_atoms == (0, 1)
    assert emb.fragments[0].siesta_input_atoms == (0, 1, 2)
    assert emb.fragments[0].siesta_buffer_atoms == (2,)
    assert emb.fragments[1].siesta_core_atoms == (2, 3)
    assert emb.fragments[1].siesta_input_atoms == (1, 2, 3)
    assert emb.fragments[1].siesta_buffer_atoms == (1,)


def test_attach_siesta_results_to_fragments_by_block_id():
    fragments = [
        type("FakeFragment", (), {"siesta_block_id": 1})(),
        type("FakeFragment", (), {"siesta_block_id": 0})(),
    ]
    results = [
        adapter.SiestaEwfResult(
            block_id=0,
            machine_id=0,
            rank=2,
            core_atom_range=(0, 1),
            input_atom_range=(0, 2),
            core_atoms=(0,),
            buffer_atoms=(1,),
            core_atom_orbital_ranges={0: (0, 2)},
            converged=True,
            total_energy_ev=-1.25,
            density_matrix_path=Path("block_0000.DM"),
            hamiltonian_matrix_path=Path("block_0000.HSX"),
            overlap_matrix_path=Path("block_0000.HSX"),
            orbital_index_path=Path("block_0000.ORB_INDX"),
            output_path=Path("siesta.out"),
            matrix_metadata={"elsi": {"solver_used": ["NTPOLY"], "last_solver_settings": {"nt_method": 2}}},
            core_matrix_metadata={"density": {"nnz": 4}},
            run_diagnostics={"num_scf_steps": 9, "convergence_reason": "scf_converged"},
        ),
        adapter.SiestaEwfResult(
            block_id=1,
            machine_id=0,
            rank=3,
            core_atom_range=(1, 2),
            input_atom_range=(0, 2),
            core_atoms=(1,),
            buffer_atoms=(0,),
            core_atom_orbital_ranges={1: (2, 4)},
            converged=True,
            total_energy_ev=-2.5,
            density_matrix_path=Path("block_0001.DM"),
            hamiltonian_matrix_path=Path("block_0001.HSX"),
            overlap_matrix_path=Path("block_0001.HSX"),
            orbital_index_path=Path("block_0001.ORB_INDX"),
            output_path=Path("siesta.out"),
            matrix_metadata={"elsi": {"solver_used": ["NTPOLY"], "last_solver_settings": {"nt_method": 2}}},
            core_matrix_metadata={"density": {"nnz": 5}},
            run_diagnostics={"num_scf_steps": 11, "convergence_reason": "scf_converged"},
        ),
    ]

    attached = adapter.attach_siesta_results_to_fragments(fragments, results)

    assert attached == fragments
    assert fragments[0].siesta_ewf_result.block_id == 1
    assert fragments[0].siesta_rank == 3
    assert fragments[0].siesta_converged is True
    assert fragments[0].siesta_total_energy_ev == -2.5
    assert fragments[0].siesta_density_matrix_path == Path("block_0001.DM")
    assert fragments[0].siesta_solver_metadata["solver_used"] == ["NTPOLY"]
    assert fragments[0].siesta_solver_metadata["last_solver_settings"]["nt_method"] == 2
    assert fragments[0].siesta_run_diagnostics["num_scf_steps"] == 11
    assert fragments[0].siesta_core_atom_orbital_ranges == {1: (2, 4)}
    assert fragments[0].siesta_core_matrix_metadata == {"density": {"nnz": 5}}
    assert fragments[1].siesta_ewf_result.block_id == 0


def test_attach_siesta_results_to_fragments_rejects_missing_result():
    fragment = type("FakeFragment", (), {"siesta_block_id": 4})()

    with pytest.raises(ValueError, match="Missing SIESTA EWF result"):
        adapter.attach_siesta_results_to_fragments([fragment], [])

    assert adapter.attach_siesta_results_to_fragments([fragment], [], strict=False) == []


def test_load_siesta_results_to_fragments_projects_run_directory(tmp_path):
    (tmp_path / "blocks.json").write_text(
        json.dumps(
            [
                {
                    "block_id": 0,
                    "core_atom_start": 0,
                    "core_atom_end": 1,
                    "input_atom_start": 0,
                    "input_atom_end": 2,
                    "local_to_global_atom_index": [0, 1],
                }
            ]
        )
    )
    (tmp_path / "results.json").write_text(
        json.dumps(
            [
                {
                    "block_id": 0,
                    "rank": 0,
                    "returncode": 0,
                    "converged": True,
                    "total_energy_ev": -3.0,
                    "density_matrix_path": "x.DM",
                    "hamiltonian_matrix_path": "x.HSX",
                    "overlap_matrix_path": "x.HSX",
                    "orbital_index_path": "x.ORB_INDX",
                    "output_path": "siesta.out",
                    "atom_orbital_ranges": {"0": [0, 2], "1": [2, 4]},
                    "matrix_metadata": {"elsi": {"solver_used": ["NTPOLY"], "last_solver_settings": {"nt_method": 2}}},
                    "run_diagnostics": {"num_scf_steps": 13, "convergence_reason": "scf_converged"},
                }
            ]
        )
    )
    (tmp_path / "predictive_ewf_closure.json").write_text(
        json.dumps(
            {
                "version": 1,
                "closure_level": "predictive-mean-field-ewf-closure-v1",
                "status": "ready",
                "uses_reference_energy": False,
                "production_predictive_physics_ready": False,
                "correlated_solver_status": "not_run_mean_field_surrogate_only",
                "bath_construction": {
                    "terms": [
                        {"block_id": 0, "core_atom": 0, "environment_atom": 1, "bath_rank": 1},
                        {"block_id": 1, "core_atom": 2, "environment_atom": 3, "bath_rank": 1},
                    ]
                },
                "double_counting": {
                    "terms": [
                        {
                            "block_id": 0,
                            "embedding_potential_expectation_ev": -0.25,
                            "matched_entries": 4,
                        }
                    ]
                },
                "energy": {
                    "predictive_total_energy_ev": -3.25,
                    "double_counting_energy_correction_ev": 0.25,
                },
                "production_blockers": ["No correlated fragment solver has been run"],
            }
        )
    )
    (tmp_path / "cluster_hamiltonians.json").write_text(
        json.dumps(
            {
                "version": 1,
                "artifact_level": "solver-ready-one-electron-cluster-v1",
                "basis_model": "core-ao-plus-boundary-density-svd-bath",
                "orthogonalization": "lowdin",
                "ready": True,
                "blocks": [
                    {
                        "block_id": 0,
                        "npz_path": str(tmp_path / "block_0000" / "cluster_hamiltonian_block_0000.npz"),
                        "ready_for_correlated_solver": True,
                        "cluster_basis_size": 2,
                    }
                ],
            }
        )
    )
    (tmp_path / "cluster_solver_results.json").write_text(
        json.dumps(
            {
                "version": 1,
                "solver_level": "one-electron-lowdin-cluster-reference-v1",
                "solver_kind": "one_electron_reference",
                "ready": True,
                "correlated_solver_status": "not_run_one_electron_reference_only",
                "blocks": [
                    {
                        "block_id": 0,
                        "solver_status": "solved",
                        "density_projected_one_electron_energy_ev": -1.25,
                        "aufbau_one_electron_energy_ev": -1.5,
                    }
                ],
            }
        )
    )
    (tmp_path / "effective_correlated_results.json").write_text(
        json.dumps(
            {
                "solver_level": "effective-interaction-second-order-cluster-v1",
                "solver_kind": "model_correlated_effective_interaction",
                "ready": True,
                "correlated_solver_status": "model_effective_interaction_solved",
                "effective_interaction_u_ev": 1.0,
                "total_correlation_energy_ev": -0.125,
                "uses_ab_initio_two_electron_integrals": False,
            }
        )
    )
    (tmp_path / "effective_correlated_results.json").write_text(
        json.dumps(
            {
                "version": 1,
                "solver_level": "effective-interaction-second-order-cluster-v1",
                "solver_kind": "model_correlated_effective_interaction",
                "ready": True,
                "correlated_solver_status": "model_effective_interaction_solved",
                "uses_ab_initio_two_electron_integrals": False,
                "effective_interaction_u_ev": 1.0,
                "blocks": [
                    {
                        "block_id": 0,
                        "solver_status": "solved",
                        "correlation_energy_ev": -0.125,
                    }
                ],
            }
        )
    )
    fragment = type("FakeFragment", (), {"siesta_block_id": 0})()

    attached = adapter.load_siesta_results_to_fragments(
        tmp_path,
        [fragment],
        require_matrices=False,
    )

    assert attached == [fragment]
    assert fragment.siesta_ewf_result.block_id == 0
    assert fragment.siesta_core_atom_orbital_ranges == {0: (0, 2)}
    assert fragment.siesta_density_matrix_path == Path("x.DM")
    assert fragment.siesta_solver_metadata["solver_used"] == ["NTPOLY"]
    assert fragment.siesta_run_diagnostics["num_scf_steps"] == 13
    assert fragment.siesta_predictive_ewf_closure_ready is True
    assert fragment.siesta_production_predictive_physics_ready is False
    assert fragment.siesta_predictive_total_energy_ev == -3.25
    assert fragment.siesta_predictive_double_counting_energy_ev == 0.25
    assert fragment.siesta_predictive_bath_terms == [
        {"block_id": 0, "core_atom": 0, "environment_atom": 1, "bath_rank": 1}
    ]
    assert fragment.siesta_embedding_potential_expectation_terms[0]["matched_entries"] == 4
    assert fragment.siesta_predictive_ewf_closure["correlated_solver_status"] == "not_run_mean_field_surrogate_only"
    assert fragment.siesta_cluster_hamiltonians_ready is True
    assert fragment.siesta_cluster_ready_for_correlated_solver is True
    assert fragment.siesta_cluster_hamiltonian_path == tmp_path / "block_0000" / "cluster_hamiltonian_block_0000.npz"
    assert fragment.siesta_cluster_hamiltonian_metadata["cluster_basis_size"] == 2
    assert fragment.siesta_cluster_solver_results_ready is True
    assert fragment.siesta_cluster_solver_status == "solved"
    assert fragment.siesta_cluster_one_electron_energy_ev == -1.25
    assert fragment.siesta_cluster_aufbau_energy_ev == -1.5
    assert fragment.siesta_effective_correlated_results_ready is True
    assert fragment.siesta_effective_correlated_solver_status == "solved"
    assert fragment.siesta_effective_correlation_energy_ev == -0.125
    assert fragment.siesta_effective_correlated_manifest["uses_ab_initio_two_electron_integrals"] is False


def test_generate_block_directories(tmp_path):
    fdf = adapter.parse_fdf(Path(__file__).resolve().parents[4] / "testcases" / "0386.fdf")
    blocks = adapter.partition_contiguous_atoms(len(fdf.atoms), block_atoms=200, buffer_atoms=10, num_machines=2)

    generated = adapter.generate_block_directories(fdf, blocks, tmp_path)

    assert len(generated) == 2
    assert (generated[0] / "input.fdf").exists()
    assert (generated[0] / "block.json").exists()
    metadata = (generated[1] / "block.json").read_text()
    assert '"core_atom_start": 200' in metadata
    assert '"input_atom_start": 190' in metadata


def test_generate_block_input_forces_required_output_files(tmp_path):
    fdf_path = tmp_path / "global.fdf"
    fdf_path.write_text(
        "\n".join(
            [
                "SystemLabel      global",
                "NumberOfAtoms    2",
                "NumberOfSpecies  1",
                "SolutionMethod   diagon",
                "SaveHS           false",
                "WriteDM          false",
                "%block AtomicCoordinatesAndAtomicSpecies",
                "0.0 0.0 0.0 1",
                "1.0 0.0 0.0 1",
                "%endblock AtomicCoordinatesAndAtomicSpecies",
            ]
        )
        + "\n"
    )
    fdf = adapter.parse_fdf(fdf_path)
    blocks = adapter.partition_contiguous_atoms(len(fdf.atoms), block_atoms=2)

    solver = adapter.SiestaSolverConfig(
        ntpoly_filter=1.0e-8,
        ntpoly_tolerance=5.0e-7,
        max_scf_iterations=180,
        dm_number_pulay=8,
        dm_mixing_weight=0.03,
    )
    generated = adapter.generate_block_directories(fdf, blocks, tmp_path / "runs", solver_config=solver)
    input_text = (generated[0] / "input.fdf").read_text()
    solver_config = json.loads((generated[0] / "solver_config.json").read_text())

    assert "SystemLabel      block_0000" in input_text
    assert "SolutionMethod     ELSI" in input_text
    assert "ELSI.Solver        ntpoly" in input_text
    assert "ELSI.NTPoly.Method 2" in input_text
    assert "ELSI.NTPoly.Filter 1.0e-08" in input_text
    assert "ELSI.NTPoly.Tolerance 5.0e-07" in input_text
    assert "MaxSCFIterations    180" in input_text
    assert "DM.NumberPulay    8" in input_text
    assert "DM.MixingWeight    0.030000" in input_text
    assert "SaveHS           true" in input_text
    assert "WriteDM          true" in input_text
    assert "WriteOrbitalIndex true" in input_text
    assert solver_config["elsi_solver"] == "ntpoly"
    assert solver_config["ntpoly_method"] == 2
    assert solver_config["max_scf_iterations"] == 180


def test_infer_bonds_and_boundary_manifest_detects_buffer_coverage(tmp_path):
    fdf_path = tmp_path / "chain.fdf"
    fdf_path.write_text(
        "\n".join(
            [
                "SystemLabel chain",
                "NumberOfAtoms 4",
                "NumberOfSpecies 2",
                "%block ChemicalSpeciesLabel",
                "1 6 C",
                "2 1 H",
                "%endblock ChemicalSpeciesLabel",
                "%block AtomicCoordinatesAndAtomicSpecies",
                "0.0 0.0 0.0 1",
                "1.5 0.0 0.0 1",
                "0.0 1.1 0.0 2",
                "1.5 1.1 0.0 2",
                "%endblock AtomicCoordinatesAndAtomicSpecies",
            ]
        )
        + "\n"
    )
    fdf = adapter.parse_fdf(fdf_path)
    blocks = [adapter.SiestaBlock(0, 0, 1, 0, 2, machine_id=0)]

    bonds = adapter.infer_bonds(fdf)
    payload = adapter.write_boundary_manifest(tmp_path, fdf, blocks)

    assert fdf.species_labels == {1: "C", 2: "H"}
    assert {(bond["atom_i"], bond["atom_j"]) for bond in bonds} == {(0, 1), (0, 2), (1, 3)}
    assert payload["num_boundary_bonds"] == 2
    assert payload["num_uncovered_boundary_bonds"] == 1
    covered = [bond for bond in payload["blocks"][0]["boundary_bonds"] if bond["covered_by_input"]]
    uncovered = [bond for bond in payload["blocks"][0]["boundary_bonds"] if not bond["covered_by_input"]]
    assert [(bond["atom_i"], bond["atom_j"]) for bond in covered] == [(0, 1)]
    assert [(bond["atom_i"], bond["atom_j"]) for bond in uncovered] == [(0, 2)]


def test_grid_bond_inference_matches_bruteforce_for_polyethylene_386(tmp_path):
    repo_root = Path(__file__).resolve().parents[4]
    completed = subprocess.run(
        [sys.executable, str(repo_root / "testcases" / "gen.py"), "128"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )
    fdf_path = tmp_path / "pe128.fdf"
    fdf_path.write_text(completed.stdout)
    fdf = adapter.parse_fdf(fdf_path)

    grid = adapter.infer_bonds(fdf)
    brute = adapter._infer_bonds_bruteforce(fdf)

    assert [(bond["atom_i"], bond["atom_j"]) for bond in grid] == [
        (bond["atom_i"], bond["atom_j"]) for bond in brute
    ]


def test_embedding_contract_manifest_records_pending_boundary_corrections(tmp_path):
    boundary_payload = {
        "blocks": [
            {
                "block_id": 0,
                "core_atom_range": [0, 1],
                "boundary_bonds": [
                    {
                        "atom_i": 0,
                        "atom_j": 1,
                        "covered_by_input": True,
                        "distance_angstrom": 1.5,
                        "species_i": "C",
                        "species_j": "C",
                    },
                    {
                        "atom_i": 0,
                        "atom_j": 2,
                        "covered_by_input": False,
                        "distance_angstrom": 1.1,
                        "species_i": "C",
                        "species_j": "H",
                    },
                ],
            }
        ]
    }
    (tmp_path / "boundary.json").write_text(json.dumps(boundary_payload))

    payload = adapter.write_embedding_contract_manifest(tmp_path)

    assert payload["embedding_level"] == "boundary-buffer-contract"
    assert payload["matrix_ownership"] == "core_owned"
    assert payload["num_terms"] == 2
    assert payload["num_pending_embedding_terms"] == 1
    assert payload["num_uncovered_boundary_terms"] == 1
    assert payload["terms"][0]["status"] == "pending_embedding_correction"
    assert payload["terms"][0]["requires_embedding_potential"] is True
    assert payload["terms"][1]["status"] == "invalid_uncovered_boundary"
    assert json.loads((tmp_path / "embedding_contract.json").read_text()) == payload


def test_boundary_corrections_manifest_creates_parameterized_minimal_closure(tmp_path):
    (tmp_path / "embedding_contract.json").write_text(
        json.dumps(
            {
                "terms": [
                    {
                        "block_id": 0,
                        "bond_atoms": [1, 2],
                        "core_atom": 1,
                        "environment_atom": 2,
                        "status": "pending_embedding_correction",
                    },
                    {
                        "block_id": 1,
                        "bond_atoms": [3, 4],
                        "core_atom": 3,
                        "environment_atom": 4,
                        "status": "invalid_uncovered_boundary",
                    },
                ]
            }
        )
    )

    payload = adapter.write_boundary_corrections_manifest(tmp_path)

    assert payload["correction_level"] == "minimal-boundary-closure"
    assert payload["closure_model"] == "core-owned-buffer-saturated-zero-shift"
    assert payload["num_corrections"] == 1
    assert payload["num_parameterized_corrections"] == 1
    assert payload["num_unparameterized_corrections"] == 0
    assert payload["corrections"][0]["correction_type"] == "boundary_bond_embedding"
    assert payload["corrections"][0]["hamiltonian_embedding_potential"]["value_ev"] == 0.0
    assert payload["corrections"][0]["energy_correction_ev"] == 0.0
    assert payload["corrections"][0]["status"] == "parameterized"
    assert json.loads((tmp_path / "boundary_corrections.json").read_text()) == payload


def test_boundary_corrections_manifest_can_write_unparameterized_slots(tmp_path):
    (tmp_path / "embedding_contract.json").write_text(
        json.dumps(
            {
                "terms": [
                    {
                        "block_id": 0,
                        "bond_atoms": [1, 2],
                        "core_atom": 1,
                        "environment_atom": 2,
                        "status": "pending_embedding_correction",
                    }
                ]
            }
        )
    )

    payload = adapter.write_boundary_corrections_manifest(tmp_path, parameterize=False)

    assert payload["correction_level"] == "placeholder"
    assert payload["num_unparameterized_corrections"] == 1
    assert payload["corrections"][0]["status"] == "not_parameterized"


def test_electron_constraint_manifest_reports_valence_deviation(tmp_path):
    fdf_path = tmp_path / "methane.fdf"
    fdf_path.write_text(
        "\n".join(
            [
                "SystemLabel methane",
                "NumberOfAtoms 2",
                "NumberOfSpecies 2",
                "%block ChemicalSpeciesLabel",
                "1 6 C",
                "2 1 H",
                "%endblock ChemicalSpeciesLabel",
                "%block AtomicCoordinatesAndAtomicSpecies",
                "0.0 0.0 0.0 1",
                "1.0 0.0 0.0 2",
                "%endblock AtomicCoordinatesAndAtomicSpecies",
            ]
        )
        + "\n"
    )
    fdf = adapter.parse_fdf(fdf_path)
    (tmp_path / "global_matrices.json").write_text(json.dumps({"density_overlap_trace_total": 4.5}))

    payload = adapter.write_electron_constraint_manifest(tmp_path, fdf)

    assert fdf.species_atomic_numbers == {1: 6, 2: 1}
    assert adapter.estimate_valence_electron_count(fdf) == 5
    assert payload["constraint_level"] == "global-electron-closure"
    assert payload["target_valence_electrons"] == 5.0
    assert payload["observed_density_overlap_trace"] == 4.5
    assert payload["electron_count_deviation"] == -0.5
    assert payload["chemical_potential_status"] == "applied"
    assert payload["electron_count_correction"] == 0.5
    assert payload["corrected_density_overlap_trace"] == 5.0
    assert payload["corrected_electron_count_deviation"] == 0.0
    assert json.loads((tmp_path / "electron_constraint.json").read_text()) == payload


def test_electron_constraint_manifest_can_remain_diagnostic(tmp_path):
    fdf_path = tmp_path / "methane.fdf"
    fdf_path.write_text(
        "\n".join(
            [
                "SystemLabel methane",
                "NumberOfAtoms 1",
                "NumberOfSpecies 1",
                "%block ChemicalSpeciesLabel",
                "1 6 C",
                "%endblock ChemicalSpeciesLabel",
                "%block AtomicCoordinatesAndAtomicSpecies",
                "0.0 0.0 0.0 1",
                "%endblock AtomicCoordinatesAndAtomicSpecies",
            ]
        )
        + "\n"
    )
    fdf = adapter.parse_fdf(fdf_path)
    (tmp_path / "global_matrices.json").write_text(json.dumps({"density_overlap_trace_total": 3.5}))

    payload = adapter.write_electron_constraint_manifest(tmp_path, fdf, apply_correction=False)

    assert payload["chemical_potential_status"] == "not_applied"
    assert payload["electron_count_correction"] is None


def test_physical_readiness_report_blocks_diagnostic_backend_only_results(tmp_path):
    (tmp_path / "validation.json").write_text(
        json.dumps(
            {
                "ok": True,
                "energy_policy": "diagnostic_block_sum_not_embedded_total",
            }
        )
    )
    (tmp_path / "embedding_contract.json").write_text(json.dumps({"num_pending_embedding_terms": 2}))
    (tmp_path / "boundary_corrections.json").write_text(json.dumps({"num_unparameterized_corrections": 2}))
    (tmp_path / "electron_constraint.json").write_text(
        json.dumps(
            {
                "chemical_potential_status": "not_applied",
                "electron_count_deviation": -0.25,
            }
        )
    )
    (tmp_path / "global_matrices.json").write_text(json.dumps({"density_overlap_trace_total": 9.75}))

    payload = adapter.write_physical_readiness_manifest(tmp_path)

    assert payload["backend_artifacts_ready"] is True
    assert payload["minimal_embedded_closure_ready"] is False
    assert payload["embedded_observable_ready"] is False
    assert payload["reference_benchmark_ready"] is False
    assert payload["predictive_embedding_ready"] is False
    assert payload["status"] == "diagnostic_backend_only"
    assert "2 boundary embedding terms" in payload["blockers"][0]
    assert "2 boundary correction slots" in payload["blockers"][1]
    assert "chemical-potential constraint is not applied" in payload["blockers"][2]
    assert payload["diagnostic_outputs"]["density_overlap_trace_total"] == 9.75
    assert payload["diagnostic_outputs"]["electron_count_deviation"] == -0.25
    assert json.loads((tmp_path / "physical_readiness.json").read_text()) == payload


def test_physical_readiness_report_allows_completed_embedding_contract(tmp_path):
    (tmp_path / "validation.json").write_text(json.dumps({"ok": True}))
    (tmp_path / "embedding_contract.json").write_text(json.dumps({"num_pending_embedding_terms": 0}))
    (tmp_path / "boundary_corrections.json").write_text(
        json.dumps(
            {
                "num_unparameterized_corrections": 0,
                "reference_total_energy_ev": -1.2,
                "total_calibrated_energy_correction_ev": -0.2,
            }
        )
    )
    (tmp_path / "electron_constraint.json").write_text(json.dumps({"chemical_potential_status": "applied"}))
    (tmp_path / "embedded_observables.json").write_text(json.dumps({"embedded_total_energy_ev": -1.0}))
    (tmp_path / "embedding_benchmark.json").write_text(
        json.dumps({"ok": False, "energy_error_ev": 0.2, "energy_error_per_atom_ev": 0.1})
    )

    payload = adapter.build_physical_readiness_report(tmp_path)

    assert payload["backend_artifacts_ready"] is True
    assert payload["minimal_embedded_closure_ready"] is True
    assert payload["embedded_observable_ready"] is True
    assert payload["reference_calibrated_correction_ready"] is False
    assert payload["benchmark_manifest_ready"] is True
    assert payload["reference_benchmark_ready"] is False
    assert payload["predictive_embedding_ready"] is False
    assert payload["status"] == "embedded_observable_ready"
    assert payload["blockers"] == []
    assert payload["diagnostic_outputs"]["benchmark_ok"] is False
    assert payload["diagnostic_outputs"]["benchmark_energy_error_ev"] == 0.2
    assert payload["diagnostic_outputs"]["boundary_reference_total_energy_ev"] == -1.2


def test_physical_readiness_reports_predictive_boundary_potential_not_self_consistent(tmp_path):
    (tmp_path / "validation.json").write_text(json.dumps({"ok": True}))
    (tmp_path / "embedding_contract.json").write_text(json.dumps({"num_pending_embedding_terms": 1}))
    (tmp_path / "boundary_corrections.json").write_text(
        json.dumps(
            {
                "correction_level": "predictive-boundary-coupling-v1",
                "num_parameterized_corrections": 1,
                "num_unparameterized_corrections": 0,
            }
        )
    )
    (tmp_path / "electron_constraint.json").write_text(json.dumps({"chemical_potential_status": "applied"}))
    (tmp_path / "embedded_observables.json").write_text(json.dumps({"embedded_total_energy_ev": -1.5}))
    (tmp_path / "predictive_embedding_potential.json").write_text(
        json.dumps(
            {
                "potential_level": "predictive-boundary-potential-v1",
                "model": "density_hamiltonian_boundary_coupling_v1",
                "uses_reference_energy": False,
                "num_parameterized_terms": 1,
                "total_predictive_energy_correction_ev": -0.5,
                "self_consistency_status": "single_shot_not_self_consistent",
                "sIESTA_external_potential_applied": False,
                "blockers": ["not injected"],
            }
        )
    )
    (tmp_path / "cluster_hamiltonians.json").write_text(
        json.dumps(
            {
                "artifact_level": "solver-ready-one-electron-cluster-v1",
                "ready": True,
                "num_written_blocks": 1,
            }
        )
    )
    (tmp_path / "cluster_solver_results.json").write_text(
        json.dumps(
            {
                "solver_level": "one-electron-lowdin-cluster-reference-v1",
                "solver_kind": "one_electron_reference",
                "ready": True,
                "num_solved_blocks": 1,
                "total_density_projected_one_electron_energy_ev": -2.0,
                "total_aufbau_one_electron_energy_ev": -2.5,
            }
        )
    )
    (tmp_path / "effective_correlated_results.json").write_text(
        json.dumps(
            {
                "solver_level": "effective-interaction-second-order-cluster-v1",
                "solver_kind": "model_correlated_effective_interaction",
                "ready": True,
                "correlated_solver_status": "model_effective_interaction_solved",
                "effective_interaction_u_ev": 1.0,
                "total_correlation_energy_ev": -0.125,
                "uses_ab_initio_two_electron_integrals": False,
            }
        )
    )
    (tmp_path / "effective_interaction_benchmark_scan.json").write_text(
        json.dumps(
            {
                "status": "scan_complete",
                "best_sample": {"effective_interaction_u_ev": 0.0, "energy_error_ev": -2.0},
                "reference_fit_possible_with_real_nonnegative_u": False,
                "reference_fit_reason": "required correction has opposite sign",
            }
        )
    )

    payload = adapter.build_physical_readiness_report(tmp_path)

    assert payload["predictive_boundary_potential_ready"] is True
    assert payload["cluster_hamiltonians_ready"] is True
    assert payload["cluster_solver_results_ready"] is True
    assert payload["effective_correlated_results_ready"] is True
    assert payload["effective_interaction_benchmark_scan_ready"] is True
    assert payload["predictive_embedding_ready"] is False
    assert payload["predictive_embedding_status"] == "single_shot_not_self_consistent"
    assert payload["diagnostic_outputs"]["predictive_potential_uses_reference_energy"] is False
    assert payload["diagnostic_outputs"]["predictive_total_energy_correction_ev"] == -0.5
    assert payload["diagnostic_outputs"]["predictive_siesta_external_potential_applied"] is False
    assert payload["diagnostic_outputs"]["cluster_hamiltonian_ready"] is True
    assert payload["diagnostic_outputs"]["cluster_hamiltonian_num_written_blocks"] == 1
    assert payload["diagnostic_outputs"]["cluster_solver_ready"] is True
    assert payload["diagnostic_outputs"]["cluster_solver_num_solved_blocks"] == 1
    assert payload["diagnostic_outputs"]["cluster_solver_total_density_projected_one_electron_energy_ev"] == -2.0
    assert payload["diagnostic_outputs"]["effective_correlated_ready"] is True
    assert payload["diagnostic_outputs"]["effective_correlation_energy_ev"] == -0.125
    assert payload["diagnostic_outputs"]["effective_uses_ab_initio_two_electron_integrals"] is False
    assert payload["diagnostic_outputs"]["effective_interaction_scan_best_u_ev"] == 0.0
    assert payload["diagnostic_outputs"]["effective_interaction_reference_fit_possible"] is False


def test_embedded_observables_manifest_combines_energy_and_electron_closure(tmp_path):
    (tmp_path / "validation.json").write_text(
        json.dumps({"total_block_energy_ev": -10.0})
    )
    (tmp_path / "boundary_corrections.json").write_text(
        json.dumps(
            {
                "closure_model": "core-owned-buffer-saturated-zero-shift",
                "corrections": [
                    {"energy_correction_ev": -0.25},
                    {"energy_correction_ev": 0.0},
                ],
            }
        )
    )
    (tmp_path / "electron_constraint.json").write_text(
        json.dumps(
            {
                "policy": "global_trace_shift_closure",
                "target_valence_electrons": 5.0,
                "observed_density_overlap_trace": 4.5,
                "corrected_density_overlap_trace": 5.0,
                "corrected_electron_count_deviation": 0.0,
            }
        )
    )
    (tmp_path / "global_matrices.json").write_text(json.dumps({"density_overlap_trace_total": 4.5}))
    (tmp_path / "cluster_solver_results.json").write_text(
        json.dumps(
            {
                "solver_level": "one-electron-lowdin-cluster-reference-v1",
                "solver_kind": "one_electron_reference",
                "ready": True,
                "total_density_projected_one_electron_energy_ev": -2.0,
                "total_aufbau_one_electron_energy_ev": -2.5,
                "correlated_solver_status": "not_run_one_electron_reference_only",
            }
        )
    )
    (tmp_path / "effective_correlated_results.json").write_text(
        json.dumps(
            {
                "solver_level": "effective-interaction-second-order-cluster-v1",
                "solver_kind": "model_correlated_effective_interaction",
                "ready": True,
                "total_correlation_energy_ev": -0.125,
                "correlated_solver_status": "model_effective_interaction_solved",
                "effective_interaction_u_ev": 1.0,
            }
        )
    )

    payload = adapter.write_embedded_observables_manifest(tmp_path)

    assert payload["observable_level"] == "minimal-embedded-closure"
    assert payload["embedded_total_energy_ev"] == -10.25
    assert payload["boundary_energy_correction_ev"] == -0.25
    assert payload["corrected_density_overlap_trace"] == 5.0
    assert payload["cluster_solver_ready"] is True
    assert payload["cluster_solver_total_density_projected_one_electron_energy_ev"] == -2.0
    assert payload["cluster_solver_correlated_status"] == "not_run_one_electron_reference_only"
    assert payload["effective_correlated_results_ready"] is True
    assert payload["effective_correlation_energy_ev"] == -0.125
    assert payload["effective_embedded_total_energy_ev"] == -10.375
    assert json.loads((tmp_path / "embedded_observables.json").read_text()) == payload


def test_embedding_benchmark_manifest_compares_reference_observables(tmp_path):
    (tmp_path / "embedded_observables.json").write_text(
        json.dumps(
            {
                "embedded_total_energy_ev": -10.25,
                "corrected_density_overlap_trace": 5.0,
            }
        )
    )
    (tmp_path / "global_matrices.json").write_text(json.dumps({"natoms": 5}))

    payload = adapter.write_embedding_benchmark_manifest(
        tmp_path,
        {
            "label": "full-siesta",
            "total_energy_ev": -10.0,
            "density_overlap_trace_total": 5.0,
        },
        energy_tolerance_ev=0.3,
    )

    assert payload["reference_label"] == "full-siesta"
    assert payload["reference_kind"] == "external_reference"
    assert payload["reference_is_external"] is True
    assert payload["status"] == "passed"
    assert payload["energy_error_ev"] == -0.25
    assert payload["energy_error_per_atom_ev"] == -0.05
    assert payload["electron_count_error"] == 0.0
    assert payload["ok"] is True
    assert json.loads((tmp_path / "embedding_benchmark.json").read_text()) == payload


def test_embedding_benchmark_manifest_flags_energy_error(tmp_path):
    (tmp_path / "embedded_observables.json").write_text(json.dumps({"embedded_total_energy_ev": -12.0}))
    (tmp_path / "global_matrices.json").write_text(json.dumps({"natoms": 2}))

    payload = adapter.build_embedding_benchmark(
        tmp_path,
        {"total_energy_ev": -10.0},
        energy_tolerance_ev=0.1,
    )

    assert payload["energy_error_ev"] == -2.0
    assert payload["status"] == "failed"
    assert payload["energy_within_tolerance"] is False
    assert payload["ok"] is False


def test_embedding_benchmark_can_use_reference_workdir(tmp_path):
    embedded = tmp_path / "embedded"
    reference = tmp_path / "reference"
    embedded.mkdir()
    reference.mkdir()
    (embedded / "embedded_observables.json").write_text(
        json.dumps({"embedded_total_energy_ev": -10.0, "corrected_density_overlap_trace": 4.0})
    )
    (embedded / "global_matrices.json").write_text(json.dumps({"natoms": 2}))
    (reference / "embedded_observables.json").write_text(
        json.dumps({"embedded_total_energy_ev": -10.1, "corrected_density_overlap_trace": 4.0})
    )

    reference_payload = adapter.reference_observables_from_workdir(reference, label="full")
    payload = adapter.write_embedding_benchmark_from_reference_workdir(
        embedded,
        reference,
        label="full",
        energy_tolerance_ev=0.2,
    )

    assert reference_payload["label"] == "full"
    assert reference_payload["reference_is_external"] is True
    assert reference_payload["total_energy_ev"] == -10.1
    assert payload["reference_label"] == "full"
    assert payload["reference_kind"] == "external_full_system_or_higher_quality_run"
    assert payload["energy_error_ev"] == 0.09999999999999964
    assert payload["ok"] is True


def test_reference_missing_embedding_benchmark_manifest_documents_gap(tmp_path):
    (tmp_path / "embedded_observables.json").write_text(
        json.dumps({"embedded_total_energy_ev": -100.0, "corrected_density_overlap_trace": 42.0})
    )
    (tmp_path / "global_matrices.json").write_text(json.dumps({"natoms": 386}))

    payload = adapter.write_reference_missing_embedding_benchmark_manifest(
        tmp_path,
        reason="full-system TRS2 reference is too expensive for this development host",
        next_steps=["run a full-system reference on a larger node"],
    )

    assert payload["status"] == "reference_missing"
    assert payload["ok"] is False
    assert payload["reference_is_external"] is False
    assert payload["natoms"] == 386
    assert payload["embedded_total_energy_ev"] == -100.0
    assert "larger node" in payload["next_validation_steps"][0]
    assert json.loads((tmp_path / "embedding_benchmark.json").read_text()) == payload


def test_calibrate_boundary_corrections_to_reference_energy(tmp_path):
    (tmp_path / "boundary_corrections.json").write_text(
        json.dumps(
            {
                "correction_level": "minimal-boundary-closure",
                "corrections": [
                    {"energy_correction_ev": 0.0, "status": "parameterized"},
                    {"energy_correction_ev": 0.0, "status": "parameterized"},
                ],
            }
        )
    )
    (tmp_path / "embedded_observables.json").write_text(json.dumps({"embedded_total_energy_ev": -12.0}))

    payload = adapter.calibrate_boundary_corrections_to_reference(tmp_path, -10.0)

    assert payload["correction_level"] == "reference-calibrated-boundary-closure"
    assert payload["closure_model"] == "reference-total-energy-matched-boundary-shift"
    assert payload["total_calibrated_energy_correction_ev"] == 2.0
    assert [item["energy_correction_ev"] for item in payload["corrections"]] == [1.0, 1.0]
    assert payload["corrections"][0]["calibration"]["reference_total_energy_ev"] == -10.0


def test_calibrate_boundary_corrections_prefers_block_energy_baseline(tmp_path):
    (tmp_path / "boundary_corrections.json").write_text(
        json.dumps(
            {
                "correction_level": "minimal-boundary-closure",
                "corrections": [
                    {"energy_correction_ev": 100.0, "status": "parameterized"},
                    {"energy_correction_ev": 100.0, "status": "parameterized"},
                ],
            }
        )
    )
    (tmp_path / "validation.json").write_text(json.dumps({"total_block_energy_ev": -12.0}))
    (tmp_path / "embedded_observables.json").write_text(json.dumps({"embedded_total_energy_ev": 188.0}))

    payload = adapter.calibrate_boundary_corrections_to_reference(tmp_path, -10.0)

    assert payload["total_calibrated_energy_correction_ev"] == 2.0
    assert [item["energy_correction_ev"] for item in payload["corrections"]] == [1.0, 1.0]
    assert payload["calibration_baseline_source"] == "validation.total_block_energy_ev"
    assert payload["corrections"][0]["calibration"]["baseline_total_energy_ev"] == -12.0


def test_predictive_boundary_corrections_use_returned_dm_hsx_without_reference(tmp_path):
    dm_path = tmp_path / "block_0000.DM"
    hsx_path = tmp_path / "block_0000.HSX"
    _write_full_dm(dm_path, [2, 1], [[1, 2], [2]], [[[1.0, 0.2], [1.0]]])
    _write_full_hsx(hsx_path, [2, 1], [[1, 2], [2]], [[[0.0, 5.0], [0.0]]], [[1.0, 0.1], [1.0]])
    (tmp_path / "embedding_contract.json").write_text(
        json.dumps(
            {
                "terms": [
                    {
                        "block_id": 0,
                        "bond_atoms": [0, 1],
                        "core_atom": 0,
                        "environment_atom": 1,
                        "status": "pending_embedding_correction",
                    }
                ]
            }
        )
    )
    (tmp_path / "results.json").write_text(
        json.dumps(
            [
                {
                    "block_id": 0,
                    "rank": 0,
                    "returncode": 0,
                    "converged": True,
                    "density_matrix_path": str(dm_path),
                    "hamiltonian_matrix_path": str(hsx_path),
                    "atom_orbital_ranges": {"0": [0, 1], "1": [1, 2]},
                    "matrix_metadata": {"elsi": {"solver_used": ["NTPOLY"], "last_solver_settings": {"nt_method": 2}}},
                }
            ]
        )
    )

    payload = adapter.write_predictive_boundary_corrections_manifest(tmp_path, damping=1.0)
    potential = json.loads((tmp_path / "predictive_embedding_potential.json").read_text())

    assert payload["correction_level"] == "predictive-boundary-coupling-v1"
    assert payload["uses_reference_energy"] is False
    assert payload["num_parameterized_corrections"] == 1
    assert payload["total_predictive_energy_correction_ev"] == -0.5
    assert payload["corrections"][0]["density_hamiltonian_coupling_ev"] == 1.0
    assert payload["corrections"][0]["energy_correction_ev"] == -0.5
    assert payload["corrections"][0]["hamiltonian_embedding_potential"]["value_ev"] == -1.0
    assert payload["corrections"][0]["sparse_hamiltonian_embedding_potential"]["num_entries"] == 1
    assert payload["corrections"][0]["sparse_hamiltonian_embedding_potential"]["entries"][0]["value_ev"] == -2.5
    assert payload["sIESTA_external_potential_applied"] is False
    assert potential["source"] == "siesta_returned_dm_hsx"
    assert potential["uses_reference_energy"] is False

    block_dir = tmp_path / "block_0000"
    block_dir.mkdir()
    (block_dir / "input.fdf").write_text("SystemLabel block_0000\n")
    (tmp_path / "blocks.json").write_text(
        json.dumps(
            [
                {
                    "block_id": 0,
                    "core_atom_start": 0,
                    "core_atom_end": 1,
                    "input_atom_start": 0,
                    "input_atom_end": 2,
                }
            ]
        )
    )

    siesta_inputs = adapter.write_siesta_embedding_potential_inputs(tmp_path)

    assert siesta_inputs["num_blocks_with_potential"] == 1
    assert siesta_inputs["model"] == "sparse-nonlocal-boundary-shift-v1"
    assert (block_dir / "ewf_embedding_potential.dat").exists()
    assert "1 2 1 -2.5000000000000000e+00" in (block_dir / "ewf_embedding_potential.dat").read_text()
    assert "EWF.Embedding.PotentialFile   ewf_embedding_potential.dat" in (block_dir / "input.fdf").read_text()


def test_predictive_ewf_closure_builds_bath_and_double_counting(tmp_path):
    dm_path = tmp_path / "block_0000.DM"
    hsx_path = tmp_path / "block_0000.HSX"
    _write_full_dm(dm_path, [2, 1], [[1, 2], [2]], [[[1.0, 0.2], [1.0]]])
    _write_full_hsx(hsx_path, [2, 1], [[1, 2], [2]], [[[0.0, 5.0], [0.0]]], [[1.0, 0.1], [1.0]])
    block_dir = tmp_path / "block_0000"
    block_dir.mkdir()
    (block_dir / "ewf_embedding_potential.dat").write_text("# row col spin value_eV\n1 2 1 -2.5\n")
    (tmp_path / "validation.json").write_text(json.dumps({"ok": True}))
    (tmp_path / "results.json").write_text(
        json.dumps(
            [
                {
                    "block_id": 0,
                    "rank": 0,
                    "returncode": 0,
                    "converged": True,
                    "density_matrix_path": str(dm_path),
                    "hamiltonian_matrix_path": str(hsx_path),
                    "atom_orbital_ranges": {"0": [0, 1], "1": [1, 2]},
                    "matrix_metadata": {"elsi": {"solver_used": ["NTPOLY"], "last_solver_settings": {"nt_method": 2}}},
                }
            ]
        )
    )
    (tmp_path / "boundary_corrections.json").write_text(
        json.dumps(
            {
                "correction_level": "predictive-boundary-coupling-v1",
                "num_parameterized_corrections": 1,
                "num_unparameterized_corrections": 0,
                "corrections": [
                    {
                        "block_id": 0,
                        "core_atom": 0,
                        "environment_atom": 1,
                        "bond_atoms": [0, 1],
                        "energy_correction_ev": -0.5,
                    }
                ],
            }
        )
    )
    (tmp_path / "predictive_embedding_potential.json").write_text(
        json.dumps(
            {
                "num_parameterized_terms": 1,
                "uses_reference_energy": False,
                "self_consistency_status": "converged",
                "sIESTA_external_potential_applied": True,
            }
        )
    )
    (tmp_path / "electron_constraint.json").write_text(json.dumps({"chemical_potential_status": "applied"}))
    (tmp_path / "embedded_observables.json").write_text(
        json.dumps({"total_block_energy_ev": -10.0, "boundary_energy_correction_ev": -0.5})
    )

    closure = adapter.write_predictive_ewf_closure_manifest(tmp_path)
    observables = adapter.write_embedded_observables_manifest(tmp_path)
    readiness = adapter.build_physical_readiness_report(tmp_path)

    assert closure["status"] == "ready"
    assert closure["uses_reference_energy"] is False
    assert closure["production_predictive_physics_ready"] is False
    assert closure["bath_construction"]["total_bath_rank"] == 1
    assert closure["bath_construction"]["terms"][0]["singular_values"] == [0.2]
    assert closure["double_counting"]["embedding_potential_expectation_ev"] == -0.5
    assert closure["energy"]["double_counting_energy_correction_ev"] == 0.5
    assert closure["energy"]["predictive_total_energy_ev"] == -10.0
    assert observables["predictive_ewf_closure_status"] == "ready"
    assert observables["bath_total_rank"] == 1
    assert observables["predictive_total_energy_ev"] == -10.0
    assert readiness["predictive_ewf_closure_ready"] is True
    assert readiness["production_predictive_physics_ready"] is False
    assert readiness["diagnostic_outputs"]["predictive_ewf_bath_total_rank"] == 1


def test_cluster_hamiltonians_write_solver_ready_npz(tmp_path):
    dm_path = tmp_path / "block_0000.DM"
    hsx_path = tmp_path / "block_0000.HSX"
    _write_full_dm(dm_path, [2, 1], [[1, 2], [2]], [[[1.0, 0.2], [1.0]]])
    _write_full_hsx(hsx_path, [2, 1], [[1, 2], [2]], [[[1.0, 0.1], [0.8]]], [[1.0, 0.0], [1.0]])
    block_dir = tmp_path / "block_0000"
    block_dir.mkdir()
    (tmp_path / "blocks.json").write_text(
        json.dumps(
            [
                {
                    "block_id": 0,
                    "core_atom_start": 0,
                    "core_atom_end": 1,
                    "input_atom_start": 0,
                    "input_atom_end": 2,
                    "local_to_global_atom_index": [0, 1],
                }
            ]
        )
    )
    (tmp_path / "results.json").write_text(
        json.dumps(
            [
                {
                    "block_id": 0,
                    "rank": 0,
                    "returncode": 0,
                    "converged": True,
                    "density_matrix_path": str(dm_path),
                    "hamiltonian_matrix_path": str(hsx_path),
                    "atom_orbital_ranges": {"0": [0, 1], "1": [1, 2]},
                }
            ]
        )
    )
    (tmp_path / "predictive_ewf_closure.json").write_text(
        json.dumps(
            {
                "status": "ready",
                "bath_construction": {
                    "terms": [
                        {
                            "block_id": 0,
                            "core_atom": 0,
                            "environment_atom": 1,
                            "bath_rank": 1,
                        }
                    ]
                },
            }
        )
    )

    payload = adapter.write_cluster_hamiltonians_manifest(tmp_path)
    block = payload["blocks"][0]

    assert payload["ready"] is True
    assert block["ready_for_correlated_solver"] is True
    assert block["num_core_orbitals"] == 1
    assert block["num_bath_orbitals"] == 1
    assert block["cluster_basis_size"] == 2
    assert block["orthogonalized_basis_size"] == 2
    assert Path(block["npz_path"]).exists()
    arrays = np.load(block["npz_path"])
    assert arrays["basis_coefficients"].shape == (2, 2)
    assert arrays["hamiltonian_orthogonalized"].shape == (1, 2, 2)
    assert arrays["density_orthogonalized"].shape == (1, 2, 2)
    assert np.allclose(arrays["overlap_orthogonalized"], np.eye(2))
    assert json.loads((block_dir / "cluster_hamiltonian_block_0000.json").read_text()) == block


def test_cluster_hamiltonian_density_projection_uses_overlap_metric(tmp_path):
    dm_path = tmp_path / "block_0000.DM"
    hsx_path = tmp_path / "block_0000.HSX"
    _write_full_dm(dm_path, [1], [[1]], [[[1.0]]])
    _write_full_hsx(hsx_path, [1], [[1]], [[[2.0]]], [[2.0]])
    block_dir = tmp_path / "block_0000"
    block_dir.mkdir()
    (tmp_path / "blocks.json").write_text(
        json.dumps(
            [
                {
                    "block_id": 0,
                    "core_atom_start": 0,
                    "core_atom_end": 1,
                    "input_atom_start": 0,
                    "input_atom_end": 1,
                    "local_to_global_atom_index": [0],
                }
            ]
        )
    )
    (tmp_path / "results.json").write_text(
        json.dumps(
            [
                {
                    "block_id": 0,
                    "rank": 0,
                    "returncode": 0,
                    "converged": True,
                    "density_matrix_path": str(dm_path),
                    "hamiltonian_matrix_path": str(hsx_path),
                    "atom_orbital_ranges": {"0": [0, 1]},
                }
            ]
        )
    )
    (tmp_path / "predictive_ewf_closure.json").write_text(json.dumps({"bath_construction": {"terms": []}}))

    payload = adapter.write_cluster_hamiltonians_manifest(tmp_path)
    arrays = np.load(payload["blocks"][0]["npz_path"])

    assert arrays["overlap_cluster"].tolist() == [[2.0]]
    assert arrays["density_cluster"].tolist() == [[[4.0]]]
    assert np.allclose(arrays["density_orthogonalized"], [[[2.0]]])


def test_cluster_solver_consumes_cluster_npz(tmp_path):
    block_dir = tmp_path / "block_0000"
    block_dir.mkdir()
    npz_path = block_dir / "cluster_hamiltonian_block_0000.npz"
    np.savez_compressed(
        npz_path,
        hamiltonian_orthogonalized=np.asarray([[[1.0, 0.0], [0.0, 3.0]]]),
        density_orthogonalized=np.asarray([[[2.0, 0.0], [0.0, 0.0]]]),
        overlap_orthogonalized=np.eye(2),
    )
    (tmp_path / "cluster_hamiltonians.json").write_text(
        json.dumps(
            {
                "num_blocks": 1,
                "ready": True,
                "blocks": [
                    {
                        "block_id": 0,
                        "npz_path": str(npz_path),
                        "cluster_basis_size": 2,
                        "orthogonalized_basis_size": 2,
                        "num_core_orbitals": 1,
                        "num_bath_orbitals": 1,
                    }
                ],
            }
        )
    )

    payload = adapter.write_cluster_solver_results_manifest(tmp_path)
    block = payload["blocks"][0]

    assert payload["ready"] is True
    assert payload["solver_kind"] == "one_electron_reference"
    assert payload["correlated_solver_status"] == "not_run_one_electron_reference_only"
    assert block["solver_status"] == "solved"
    assert block["electron_count_from_density"] == 2.0
    assert block["density_projected_one_electron_energy_ev"] == 2.0
    assert block["aufbau_one_electron_energy_ev"] == 2.0
    assert block["density_vs_aufbau_energy_delta_ev"] == 0.0
    assert block["spin_channels"][0]["first_eigenvalues_ev"] == [1.0, 3.0]
    assert json.loads((tmp_path / "cluster_solver_results.json").read_text()) == payload


def test_effective_interaction_solver_adds_model_correlation(tmp_path):
    block_dir = tmp_path / "block_0000"
    block_dir.mkdir()
    npz_path = block_dir / "cluster_hamiltonian_block_0000.npz"
    rotation = np.asarray([[1.0, 1.0], [1.0, -1.0]]) / 2.0**0.5
    h = rotation @ np.diag([1.0, 3.0]) @ rotation.T
    np.savez_compressed(
        npz_path,
        hamiltonian_orthogonalized=np.asarray([h]),
        density_orthogonalized=np.asarray([[[1.0, 0.0], [0.0, 1.0]]]),
        overlap_orthogonalized=np.eye(2),
    )
    (tmp_path / "cluster_hamiltonians.json").write_text(
        json.dumps(
            {
                "num_blocks": 1,
                "ready": True,
                "blocks": [
                    {
                        "block_id": 0,
                        "npz_path": str(npz_path),
                        "cluster_basis_size": 2,
                        "orthogonalized_basis_size": 2,
                    }
                ],
            }
        )
    )
    (tmp_path / "cluster_solver_results.json").write_text(
        json.dumps({"solver_level": "one-electron-lowdin-cluster-reference-v1"})
    )

    payload = adapter.write_effective_correlated_results_manifest(
        tmp_path,
        effective_interaction_u_ev=1.0,
        denominator_shift_ev=0.0,
    )
    block = payload["blocks"][0]

    assert payload["ready"] is True
    assert payload["solver_kind"] == "model_correlated_effective_interaction"
    assert payload["uses_ab_initio_two_electron_integrals"] is False
    assert payload["correlated_solver_status"] == "model_effective_interaction_solved"
    assert block["solver_status"] == "solved"
    assert block["correlation_energy_ev"] == pytest.approx(-0.125)
    assert payload["total_correlation_energy_ev"] == pytest.approx(-0.125)
    assert block["spin_channels"][0]["pair_terms_sample"][0]["coupling_ev"] == pytest.approx(0.5)
    assert json.loads((tmp_path / "effective_correlated_results.json").read_text()) == payload


def test_effective_interaction_solver_consumes_external_two_electron_integrals(tmp_path):
    block_dir = tmp_path / "block_0000"
    block_dir.mkdir()
    npz_path = block_dir / "cluster_hamiltonian_block_0000.npz"
    integral_path = block_dir / "cluster_two_electron_integrals_block_0000.npz"
    np.savez_compressed(
        npz_path,
        hamiltonian_orthogonalized=np.asarray([[[1.0, 0.0], [0.0, 3.0]]]),
        density_orthogonalized=np.asarray([[[2.0, 0.0], [0.0, 0.0]]]),
        overlap_orthogonalized=np.eye(2),
    )
    np.savez_compressed(
        integral_path,
        ovov=np.asarray([[0.25, 0.75]]),
        format=np.asarray("cluster-eigenbasis-ovov-v1"),
    )
    (tmp_path / "cluster_hamiltonians.json").write_text(
        json.dumps(
            {
                "num_blocks": 1,
                "ready": True,
                "blocks": [
                    {
                        "block_id": 0,
                        "npz_path": str(npz_path),
                        "two_electron_integrals_npz_path": str(integral_path),
                        "cluster_basis_size": 2,
                        "orthogonalized_basis_size": 2,
                    }
                ],
            }
        )
    )
    (tmp_path / "cluster_solver_results.json").write_text(
        json.dumps({"solver_level": "one-electron-lowdin-cluster-reference-v1"})
    )

    payload = adapter.write_effective_correlated_results_manifest(
        tmp_path,
        effective_interaction_u_ev=99.0,
        denominator_shift_ev=0.0,
    )
    block = payload["blocks"][0]

    assert payload["ready"] is True
    assert payload["solver_kind"] == "ab_initio_effective_interaction_second_order"
    assert payload["interaction_model"] == "external-cluster-two-electron-ovov"
    assert payload["uses_ab_initio_two_electron_integrals"] is True
    assert payload["correlated_solver_status"] == "ab_initio_effective_interaction_solved"
    assert "Effective interaction U is model supplied" not in payload["production_blockers"]
    assert block["uses_ab_initio_two_electron_integrals"] is True
    assert block["two_electron_integrals_npz_path"] == str(integral_path)
    assert block["spin_channels"][0]["pair_terms_sample"][0]["coupling_ev"] == pytest.approx(0.75)
    assert block["correlation_energy_ev"] == pytest.approx(-(0.75**2) / 2.0)
    assert payload["total_correlation_energy_ev"] == pytest.approx(-(0.75**2) / 2.0)


def test_effective_interaction_benchmark_scan_reports_fit_direction(tmp_path):
    block_dir = tmp_path / "block_0000"
    block_dir.mkdir()
    npz_path = block_dir / "cluster_hamiltonian_block_0000.npz"
    rotation = np.asarray([[1.0, 1.0], [1.0, -1.0]]) / 2.0**0.5
    h = rotation @ np.diag([1.0, 3.0]) @ rotation.T
    np.savez_compressed(
        npz_path,
        hamiltonian_orthogonalized=np.asarray([h]),
        density_orthogonalized=np.asarray([[[1.0, 0.0], [0.0, 1.0]]]),
        overlap_orthogonalized=np.eye(2),
    )
    (tmp_path / "cluster_hamiltonians.json").write_text(
        json.dumps(
            {
                "num_blocks": 1,
                "ready": True,
                "blocks": [
                    {
                        "block_id": 0,
                        "npz_path": str(npz_path),
                        "cluster_basis_size": 2,
                        "orthogonalized_basis_size": 2,
                    }
                ],
            }
        )
    )
    (tmp_path / "embedded_observables.json").write_text(json.dumps({"embedded_total_energy_ev": -10.0}))

    payload = adapter.write_effective_interaction_benchmark_scan_manifest(
        tmp_path,
        {"label": "ref", "total_energy_ev": -8.0},
        u_values_ev=[0.0, 1.0, 2.0],
        denominator_shift_ev=0.0,
    )

    assert payload["baseline_energy_error_ev"] == -2.0
    assert payload["samples"][1]["total_correlation_energy_ev"] == pytest.approx(-0.125)
    assert payload["samples"][2]["total_correlation_energy_ev"] == pytest.approx(-0.5)
    assert payload["best_sample"]["effective_interaction_u_ev"] == 0.0
    assert payload["reference_required_correlation_energy_ev"] == 2.0
    assert payload["reference_fit_possible_with_real_nonnegative_u"] is False
    assert "opposite sign" in payload["reference_fit_reason"]
    assert json.loads((tmp_path / "effective_interaction_benchmark_scan.json").read_text()) == payload


def test_read_run_config_from_environment(tmp_path):
    config = adapter.read_run_config(
        {
            "EWF_NUM_MACHINES": "4",
            "EWF_PROCS_PER_MACHINE": "16",
            "EWF_THREADS_PER_PROC": "36",
            "EWF_WORKDIR": str(tmp_path),
            "EWF_SIESTA_BIN": "/opt/siesta",
            "EWF_BLOCK_ATOMS": "1000",
            "EWF_BLOCK_BUFFER_ATOMS": "100",
            "EWF_SIESTA_DRY_RUN": "false",
            "EWF_NTPOLY_FILTER": "1e-8",
            "EWF_NTPOLY_TOLERANCE": "1e-5",
            "EWF_MAX_SCF_ITERATIONS": "180",
            "EWF_DM_NUMBER_PULAY": "8",
            "EWF_DM_MIXING_WEIGHT": "0.03",
            "EWF_PREDICTIVE_BOUNDARY": "true",
            "EWF_PREDICTIVE_BOUNDARY_DAMPING": "0.25",
            "EWF_PREDICTIVE_BOUNDARY_RERUN": "true",
            "EWF_EFFECTIVE_INTERACTION_U_EV": "1.5",
            "EWF_EFFECTIVE_INTERACTION_DENOMINATOR_SHIFT_EV": "0.02",
        }
    )

    assert config.num_machines == 4
    assert config.procs_per_machine == 16
    assert config.threads_per_proc == 36
    assert config.workdir == tmp_path
    assert config.siesta_bin == "/opt/siesta"
    assert config.block_atoms == 1000
    assert config.buffer_atoms == 100
    assert config.block_groups is None
    assert config.group_size_atoms is None
    assert config.buffer_groups == 0
    assert config.dry_run is False
    assert config.predictive_boundary is True
    assert config.predictive_boundary_damping == 0.25
    assert config.predictive_boundary_rerun is True
    assert config.effective_interaction_u_ev == 1.5
    assert config.effective_interaction_denominator_shift_ev == 0.02
    assert config.solver.ntpoly_method == 2
    assert config.solver.ntpoly_filter == 1.0e-8
    assert config.solver.ntpoly_tolerance == 1.0e-5
    assert config.solver.max_scf_iterations == 180
    assert config.solver.dm_number_pulay == 8
    assert config.solver.dm_mixing_weight == 0.03


def test_embedding_rerun_delta_manifest_records_energy_and_applied_diagnostics(tmp_path):
    (tmp_path / "first_pass_results.json").write_text(
        json.dumps(
            [
                {
                    "block_id": 0,
                    "converged": True,
                    "total_energy_ev": -10.0,
                    "wall_time_seconds": 2.0,
                    "run_diagnostics": {"num_scf_steps": 5},
                }
            ]
        )
    )
    (tmp_path / "predictive_rerun_results.json").write_text(
        json.dumps(
            [
                {
                    "block_id": 0,
                    "converged": True,
                    "total_energy_ev": -9.5,
                    "wall_time_seconds": 3.25,
                    "run_diagnostics": {"num_scf_steps": 7},
                    "matrix_metadata": {
                        "ewf_embedding_potential_applied": {
                            "applied_count": 2,
                            "skipped_count": 0,
                            "sum_value_ev": 0.5,
                        }
                    },
                }
            ]
        )
    )

    payload = adapter.write_embedding_rerun_delta_manifest(tmp_path)

    assert payload["all_rerun_blocks_converged"] is True
    assert payload["all_blocks_have_embedding_potential_applied"] is True
    assert payload["total_delta_energy_ev"] == 0.5
    assert payload["blocks"][0]["delta_wall_time_seconds"] == 1.25
    assert payload["blocks"][0]["delta_scf_steps"] == 2
    assert payload["blocks"][0]["embedding_potential_applied_diagnostics"]["applied_count"] == 2


def test_read_run_config_supports_group_partitioning(tmp_path):
    config = adapter.read_run_config(
        {
            "EWF_WORKDIR": str(tmp_path),
            "EWF_GROUP_SIZE_ATOMS": "6",
            "EWF_BLOCK_GROUPS": "10",
            "EWF_BLOCK_BUFFER_GROUPS": "2",
            "EWF_TERMINAL_CAP_ATOMS": "2",
        }
    )

    assert config.group_size_atoms == 6
    assert config.block_groups == 10
    assert config.buffer_groups == 2
    assert config.terminal_cap_atoms == 2


def test_run_assigned_blocks_sets_thread_environment(tmp_path):
    block_dir = tmp_path / "block_0000"
    block_dir.mkdir()
    runner = tmp_path / "fake-siesta.py"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import os",
                "print('OMP=' + os.environ['OMP_NUM_THREADS'])",
                "print('MKL=' + os.environ['MKL_NUM_THREADS'])",
                "print('OPENBLAS=' + os.environ['OPENBLAS_NUM_THREADS'])",
            ]
        )
        + "\n"
    )
    runner.chmod(0o755)
    config = adapter.SiestaRunConfig(
        num_machines=1,
        procs_per_machine=1,
        threads_per_proc=7,
        workdir=tmp_path,
        siesta_bin=str(runner),
        block_atoms=None,
        buffer_atoms=0,
        block_groups=None,
        group_size_atoms=None,
        buffer_groups=0,
        terminal_cap_atoms=0,
        dry_run=False,
    )

    results = adapter.run_assigned_blocks([block_dir], config)

    assert results[0].returncode == 0
    assert "OMP=7" in (block_dir / "siesta.out").read_text()


def test_run_assigned_blocks_strips_parent_mpi_environment(tmp_path, monkeypatch):
    block_dir = tmp_path / "block_0000"
    block_dir.mkdir()
    runner = tmp_path / "fake-siesta.py"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import os",
                "for name in ('OMPI_COMM_WORLD_RANK', 'PMI_RANK', 'PMIX_NAMESPACE'):",
                "    print(name + '=' + str(name in os.environ))",
                "print('OMP=' + os.environ['OMP_NUM_THREADS'])",
            ]
        )
        + "\n"
    )
    runner.chmod(0o755)
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    monkeypatch.setenv("PMI_RANK", "0")
    monkeypatch.setenv("PMIX_NAMESPACE", "test")
    config = adapter.SiestaRunConfig(
        num_machines=1,
        procs_per_machine=1,
        threads_per_proc=3,
        workdir=tmp_path,
        siesta_bin=str(runner),
        block_atoms=None,
        buffer_atoms=0,
        block_groups=None,
        group_size_atoms=None,
        buffer_groups=0,
        terminal_cap_atoms=0,
        dry_run=False,
    )

    results = adapter.run_assigned_blocks([block_dir], config)

    assert results[0].returncode == 0
    output = (block_dir / "siesta.out").read_text()
    assert "OMPI_COMM_WORLD_RANK=False" in output
    assert "PMI_RANK=False" in output
    assert "PMIX_NAMESPACE=False" in output
    assert "OMP=3" in output


def test_run_assigned_blocks_requires_binary_when_not_dry_run(tmp_path):
    config = adapter.SiestaRunConfig(
        num_machines=1,
        procs_per_machine=1,
        threads_per_proc=1,
        workdir=tmp_path,
        siesta_bin=None,
        block_atoms=None,
        buffer_atoms=0,
        block_groups=None,
        group_size_atoms=None,
        buffer_groups=0,
        terminal_cap_atoms=0,
        dry_run=False,
    )

    with pytest.raises(ValueError, match="EWF_SIESTA_BIN"):
        adapter.run_assigned_blocks([], config)


def test_read_siesta_output_with_scalar_and_matrix_paths(tmp_path):
    block_dir = tmp_path / "block_0000"
    block_dir.mkdir()
    (block_dir / "block.json").write_text(json.dumps({"block_id": 3}))
    (block_dir / "siesta.out").write_text(
        "\n".join(
            [
                "Begin run",
                "SCF converged",
                "siesta: E_KS(eV) = -123.456",
                "timer: Elapsed wall time (sec) =       1.250",
            ]
        )
    )
    _write_minimal_dm(block_dir / "block_0003.DM", norbitals=3, nspin=1)
    _write_minimal_hsx(block_dir / "block_0003.HSX", natoms=2, norbitals=3, nspin=1)
    (block_dir / "elsi_log.json").write_text(
        json.dumps(
            [
                {
                    "solver_chosen": "NTPOLY",
                    "solver_used": "NTPOLY",
                    "solver_settings": {"nt_method": 2},
                }
            ]
        )
    )
    (block_dir / "block_0003.ORB_INDX").write_text(
        "\n".join(
            [
                "io ia is spec iao n l m z p sym rc isc iuo",
                "1 1 1 C 1 2 0 0 1 F s 4.139 0 0 0 1",
                "2 1 1 C 2 2 0 0 2 F s 2.940 0 0 0 2",
                "3 2 2 H 1 1 0 0 1 F s 4.743 0 0 0 3",
            ]
        )
    )

    result = adapter.read_siesta_output(block_dir)

    assert result.block_id == 3
    assert result.converged is True
    assert result.total_energy_ev == -123.456
    assert result.wall_time_seconds == 1.25
    assert result.density_matrix_path == block_dir / "block_0003.DM"
    assert result.hamiltonian_matrix_path == block_dir / "block_0003.HSX"
    assert result.overlap_matrix_path == block_dir / "block_0003.HSX"
    assert result.orbital_index_path == block_dir / "block_0003.ORB_INDX"
    assert result.atom_orbital_ranges == {}
    assert result.to_metadata()["density_matrix_path"].endswith("block_0003.DM")
    assert result.to_metadata()["wall_time_seconds"] == 1.25
    assert result.matrix_metadata["density"]["norbitals"] == 3
    assert result.matrix_metadata["hamiltonian_overlap"]["natoms"] == 2
    assert result.matrix_metadata["hamiltonian_overlap"]["double_precision"] is True
    assert result.matrix_metadata["elsi"]["solver_used"] == ["NTPOLY"]
    assert result.matrix_metadata["elsi"]["complete"] is True
    assert result.matrix_metadata["elsi"]["last_solver_settings"]["nt_method"] == 2
    assert result.run_diagnostics["num_scf_steps"] == 0
    assert result.run_diagnostics["convergence_reason"] == "scf_converged"


def test_read_siesta_output_parses_scf_energy_and_orbital_ranges(tmp_path):
    block_dir = tmp_path / "block_0000"
    block_dir.mkdir()
    (block_dir / "block.json").write_text(json.dumps({"block_id": 0, "local_to_global_atom_index": [10, 11]}))
    (block_dir / "siesta.out").write_text(
        "\n".join(
            [
                "scf:    1    -10.0    -20.0    -20.0  0.1 0.2 0.3",
                "SCF_NOT_CONV: SCF did not converge in maximum number of steps (required).",
            ]
        )
    )
    (block_dir / "block_0000.ORB_INDX").write_text(
        "\n".join(
            [
                "io ia is spec iao n l m z p sym rc isc iuo",
                "1 1 1 C 1 2 0 0 1 F s 4.139 0 0 0 1",
                "2 1 1 C 2 2 0 0 2 F s 2.940 0 0 0 2",
                "3 2 2 H 1 1 0 0 1 F s 4.743 0 0 0 3",
            ]
        )
    )

    result = adapter.read_siesta_output(block_dir)

    assert result.converged is False
    assert result.total_energy_ev == -20.0
    assert result.run_diagnostics["num_scf_steps"] == 1
    assert result.run_diagnostics["last_scf_step"] == 1
    assert result.run_diagnostics["last_scf_energy_ev"] == -20.0
    assert result.run_diagnostics["convergence_reason"] == "max_scf_iterations_or_required_convergence_not_met"
    assert result.atom_orbital_ranges == {10: (0, 2), 11: (2, 3)}
    assert result.to_metadata()["atom_orbital_ranges"] == {"10": [0, 2], "11": [2, 3]}


def test_read_siesta_output_salvages_truncated_elsi_log(tmp_path):
    block_dir = tmp_path / "block_0000"
    block_dir.mkdir()
    (block_dir / "block.json").write_text(json.dumps({"block_id": 0}))
    (block_dir / "siesta.out").write_text(
        "\n".join(
            [
                "scf:    1    -10.0    -20.0    -20.0  0.1 0.2 0.3",
                "scf:    2    -11.0    -21.0    -21.0  0.1 0.2 0.3",
            ]
        )
    )
    (block_dir / "elsi_log.json").write_text(
        "[\n"
        + json.dumps(
            {
                "solver_chosen": "NTPOLY",
                "solver_used": "NTPOLY",
                "solver_settings": {"nt_method": 2},
            }
        )
        + ",\n"
        + '{"solver_chosen": "NTPOLY"'
    )

    result = adapter.read_siesta_output(block_dir)

    elsi = result.matrix_metadata["elsi"]
    assert elsi["complete"] is False
    assert elsi["num_records"] == 1
    assert elsi["solver_used"] == ["NTPOLY"]
    assert elsi["last_solver_settings"]["nt_method"] == 2
    assert "parse_error" in elsi
    assert result.run_diagnostics["num_scf_steps"] == 2


def test_read_hsx_sparse_reads_rows_columns_hamiltonian_and_overlap(tmp_path):
    hsx_path = tmp_path / "test.HSX"
    _write_full_hsx(
        hsx_path,
        numh=[2, 1, 2],
        columns=[[1, 3], [2], [1, 3]],
        hamiltonian_rows=[[[1.0, 0.3], [2.0], [0.3, 3.0]]],
        overlap_rows=[[1.0, 0.1], [1.0], [0.1, 1.0]],
        natoms=2,
    )

    hsx = adapter.read_hsx_sparse(hsx_path)

    assert hsx.metadata.norbitals == 3
    assert hsx.metadata.nspin == 1
    assert hsx.nnz == 5
    assert hsx.rows.tolist() == [0, 0, 1, 2, 2]
    assert hsx.cols.tolist() == [0, 2, 1, 0, 2]
    assert hsx.hamiltonian.shape == (1, 5)
    assert hsx.hamiltonian[0].tolist() == [1.0, 0.3, 2.0, 0.3, 3.0]
    assert hsx.overlap.tolist() == [1.0, 0.1, 1.0, 0.1, 1.0]

    core = hsx.core_block({0: (0, 1), 2: (2, 3)})

    assert core.to_metadata() == {"norbitals": 2, "nnz": 4, "orbital_start": 0, "orbital_end": 3}
    assert core.rows.tolist() == [0, 0, 2, 2]
    assert core.cols.tolist() == [0, 2, 0, 2]


def test_read_density_matrix_sparse_reads_rows_columns_and_density(tmp_path):
    dm_path = tmp_path / "test.DM"
    _write_full_dm(
        dm_path,
        numh=[2, 1, 2],
        columns=[[1, 3], [2], [1, 3]],
        density_rows=[[[0.8, 0.05], [0.6], [0.05, 0.7]]],
    )

    dm = adapter.read_density_matrix_sparse(dm_path)

    assert dm.metadata.norbitals == 3
    assert dm.metadata.nspin == 1
    assert dm.nnz == 5
    assert dm.rows.tolist() == [0, 0, 1, 2, 2]
    assert dm.cols.tolist() == [0, 2, 1, 0, 2]
    assert dm.density.shape == (1, 5)
    assert dm.density[0].tolist() == [0.8, 0.05, 0.6, 0.05, 0.7]
    assert dm.core_block({0: (0, 1), 2: (2, 3)}).to_metadata() == {
        "kind": "density",
        "norbitals": 2,
        "nnz": 4,
        "orbital_start": 0,
        "orbital_end": 3,
    }


def test_write_results_manifest_collects_rank_results_in_block_order(tmp_path):
    (tmp_path / "result_rank_0001.json").write_text(json.dumps([{"block_id": 3}, {"block_id": 1}]))
    (tmp_path / "result_rank_0000.json").write_text(json.dumps([{"block_id": 2, "rank": 10}]))

    payload = adapter.write_results_manifest(tmp_path)

    assert payload == [{"block_id": 1, "rank": 1}, {"block_id": 2, "rank": 10}, {"block_id": 3, "rank": 1}]
    assert json.loads((tmp_path / "results.json").read_text()) == payload


def test_write_run_summary_manifest_reports_scaling_metrics(tmp_path):
    (tmp_path / "blocks.json").write_text(
        json.dumps(
            [
                {
                    "block_id": 0,
                    "machine_id": 0,
                    "core_atom_start": 0,
                    "core_atom_end": 6,
                    "input_atom_start": 0,
                    "input_atom_end": 12,
                },
                {
                    "block_id": 1,
                    "machine_id": 1,
                    "core_atom_start": 6,
                    "core_atom_end": 14,
                    "input_atom_start": 0,
                    "input_atom_end": 14,
                },
            ]
        )
    )
    (tmp_path / "results.json").write_text(
        json.dumps(
            [
                {
                    "block_id": 0,
                    "rank": 0,
                    "returncode": 0,
                    "converged": True,
                    "wall_time_seconds": 2.0,
                    "total_energy_ev": -10.0,
                    "matrix_metadata": {
                        "density": {"norbitals": 40},
                        "hamiltonian_overlap": {"norbitals": 40},
                        "elsi": {
                            "solver_used": ["NTPOLY"],
                            "last_solver_settings": {"nt_method": 2, "nt_filter": 1.0e-9, "nt_tol": 1.0e-6},
                        },
                    },
                    "run_diagnostics": {
                        "num_scf_steps": 7,
                        "last_scf_step": 7,
                        "last_scf_energy_ev": -10.0,
                        "convergence_reason": "scf_converged",
                    },
                },
                {
                    "block_id": 1,
                    "rank": 1,
                    "returncode": 0,
                    "converged": True,
                    "wall_time_seconds": 4.0,
                    "total_energy_ev": -11.0,
                    "matrix_metadata": {
                        "density": {"norbitals": 60},
                        "hamiltonian_overlap": {"norbitals": 60},
                        "elsi": {
                            "solver_used": ["NTPOLY"],
                            "last_solver_settings": {"nt_method": 2, "nt_filter": 1.0e-9, "nt_tol": 1.0e-6},
                        },
                    },
                    "run_diagnostics": {
                        "num_scf_steps": 9,
                        "last_scf_step": 9,
                        "last_scf_energy_ev": -11.0,
                        "convergence_reason": "scf_converged",
                    },
                },
            ]
        )
    )

    payload = adapter.write_run_summary_manifest(tmp_path)

    assert payload["num_blocks"] == 2
    assert payload["num_results"] == 2
    assert payload["num_successful_results"] == 2
    assert payload["num_failed_results"] == 0
    assert payload["num_converged_results"] == 2
    assert payload["success_rate"] == 1.0
    assert payload["converged_rate"] == 1.0
    assert payload["num_ranks_with_results"] == 2
    assert payload["num_scheduled_ranks"] == 0
    assert payload["machines"] == [0, 1]
    assert payload["total_wall_time_seconds"] == 6.0
    assert payload["max_block_wall_time_seconds"] == 4.0
    assert payload["max_block_wall_time"] == 4.0
    assert payload["mean_block_wall_time_seconds"] == 3.0
    assert payload["mean_block_wall_time"] == 3.0
    assert payload["weak_scaling_efficiency_vs_baseline"] == 1.0
    assert payload["solver_used"] == ["NTPOLY"]
    assert payload["ntpoly_methods"] == [2]
    assert payload["max_scf_steps"] == 9
    assert payload["blocks"][0]["buffer_atoms"] == 6
    assert payload["blocks"][1]["density_norbitals"] == 60
    assert payload["blocks"][1]["solver_used"] == ["NTPOLY"]
    assert payload["blocks"][1]["ntpoly_method"] == 2
    assert payload["blocks"][1]["num_scf_steps"] == 9
    assert payload["blocks"][1]["convergence_reason"] == "scf_converged"
    assert json.loads((tmp_path / "run_summary.json").read_text()) == payload


def test_write_matrix_shape_report_records_per_rank_mnk_and_partition_quality(tmp_path):
    blocks = adapter.partition_contiguous_atoms(20, block_atoms=10, buffer_atoms=2, num_machines=2)
    adapter.write_schedule_manifest(tmp_path, blocks, num_machines=2, procs_per_machine=1, threads_per_proc=3)
    (tmp_path / "blocks.json").write_text(
        json.dumps([block.to_metadata(list(range(block.input_atom_start, block.input_atom_end))) for block in blocks])
    )
    (tmp_path / "results.json").write_text(
        json.dumps(
            [
                {
                    "block_id": 0,
                    "rank": 0,
                    "returncode": 0,
                    "converged": True,
                    "wall_time_seconds": 2.0,
                    "matrix_metadata": {
                        "density": {"norbitals": 60},
                        "hamiltonian_overlap": {"norbitals": 60},
                        "elsi": {"solver_used": ["NTPOLY"], "last_solver_settings": {"nt_method": 2}},
                    },
                    "run_diagnostics": {"num_scf_steps": 8},
                },
                {
                    "block_id": 1,
                    "rank": 1,
                    "returncode": 0,
                    "converged": True,
                    "wall_time_seconds": 3.0,
                    "matrix_metadata": {
                        "density": {"norbitals": 60},
                        "hamiltonian_overlap": {"norbitals": 60},
                        "elsi": {"solver_used": ["NTPOLY"], "last_solver_settings": {"nt_method": 2}},
                    },
                    "run_diagnostics": {"num_scf_steps": 9},
                },
            ]
        )
    )
    (tmp_path / "ewf_results.json").write_text(
        json.dumps(
            [
                {"block_id": 0, "core_matrix_metadata": {"density": {"norbitals": 50, "nnz": 200}}},
                {"block_id": 1, "core_matrix_metadata": {"density": {"norbitals": 50, "nnz": 210}}},
            ]
        )
    )
    (tmp_path / "global_matrices.json").write_text(json.dumps({"norbitals": 100, "nnz": 410}))

    payload = adapter.write_matrix_shape_report_manifest(tmp_path)

    assert payload["global_core_norbitals"] == 100
    assert payload["max_local_norbitals"] == 60
    assert payload["local_vs_global_norbital_ratio"] == 0.6
    assert payload["local_balance_ratio_max_over_mean"] == 1.0
    assert payload["effective_partition"] == "effective_and_balanced"
    assert payload["blocks"][0]["local_matrix"]["m"] == 60
    assert payload["blocks"][0]["local_matrix"]["n"] == 60
    assert payload["blocks"][0]["local_matrix"]["k"] == 60
    assert payload["blocks"][0]["core_matrix"]["m"] == 50
    assert payload["blocks"][0]["buffer_orbital_amplification"] == 1.2
    assert payload["blocks"][1]["threads_per_proc"] == 3
    assert payload["ranks"][0]["max_local_m"] == 60
    assert payload["ranks"][0]["sum_dense_equivalent_gemm_flops"] == 432000
    assert json.loads((tmp_path / "matrix_shape_report.json").read_text()) == payload


def test_matrix_shape_report_reads_local_sparse_nnz_from_matrix_files(tmp_path):
    dm_path = tmp_path / "block_0000.DM"
    hsx_path = tmp_path / "block_0000.HSX"
    _write_full_dm(dm_path, [2, 1], [[1, 2], [2]], [[[1.0, 0.1], [2.0]]])
    _write_full_hsx(hsx_path, [2, 1], [[1, 2], [2]], [[[1.0, 0.1], [2.0]]], [[1.0, 0.0], [1.0]])
    blocks = adapter.partition_contiguous_atoms(2, block_atoms=2)
    adapter.write_schedule_manifest(tmp_path, blocks, num_machines=1, procs_per_machine=1)
    (tmp_path / "blocks.json").write_text(
        json.dumps([blocks[0].to_metadata([0, 1])])
    )
    (tmp_path / "results.json").write_text(
        json.dumps(
            [
                {
                    "block_id": 0,
                    "rank": 0,
                    "returncode": 0,
                    "converged": True,
                    "density_matrix_path": str(dm_path),
                    "hamiltonian_matrix_path": str(hsx_path),
                    "matrix_metadata": {
                        "density": {"norbitals": 2},
                        "hamiltonian_overlap": {"norbitals": 2},
                    },
                }
            ]
        )
    )

    payload = adapter.write_matrix_shape_report_manifest(tmp_path)

    assert payload["blocks"][0]["local_matrix"]["m"] == 2
    assert payload["blocks"][0]["local_matrix"]["nnz"] == 3
    assert payload["blocks"][0]["local_matrix"]["sparse_fill_fraction"] == 0.75
    assert payload["ranks"][0]["sum_local_sparse_nnz"] == 3


def test_run_summary_uses_schedule_rank_when_results_are_missing(tmp_path):
    blocks = adapter.partition_contiguous_atoms(12, block_atoms=4, buffer_atoms=0, num_machines=2)
    adapter.write_schedule_manifest(tmp_path, blocks, num_machines=2, procs_per_machine=1)
    (tmp_path / "blocks.json").write_text(
        json.dumps([block.to_metadata(list(range(block.input_atom_start, block.input_atom_end))) for block in blocks])
    )
    (tmp_path / "results.json").write_text("[]")

    payload = adapter.write_run_summary_manifest(tmp_path)

    assert payload["scheduled_ranks"] == [0, 1]
    assert payload["success_rate"] == 0.0
    assert payload["converged_rate"] == 0.0
    assert payload["num_ranks_with_results"] == 0
    assert [block["rank"] for block in payload["blocks"]] == [0, 1, 0]


def test_write_weak_scaling_report_compares_run_summaries(tmp_path):
    run0 = tmp_path / "run0"
    run1 = tmp_path / "run1"
    run0.mkdir()
    run1.mkdir()
    (run0 / "run_summary.json").write_text(
        json.dumps(
            {
                "workdir": str(run0),
                "num_blocks": 1,
                "num_scheduled_ranks": 1,
                "num_ranks_with_results": 1,
                "num_machines": 1,
                "num_successful_results": 1,
                "num_failed_results": 0,
                "num_converged_results": 1,
                "max_block_wall_time_seconds": 10.0,
                "mean_block_wall_time_seconds": 10.0,
                "solver_used": ["NTPOLY"],
                "ntpoly_methods": [2],
                "max_scf_steps": 8,
            }
        )
    )
    (run1 / "run_summary.json").write_text(
        json.dumps(
            {
                "workdir": str(run1),
                "num_blocks": 2,
                "num_scheduled_ranks": 2,
                "num_ranks_with_results": 2,
                "num_machines": 1,
                "num_successful_results": 2,
                "num_failed_results": 0,
                "num_converged_results": 2,
                "max_block_wall_time_seconds": 12.5,
                "mean_block_wall_time_seconds": 11.0,
                "solver_used": ["NTPOLY"],
                "ntpoly_methods": [2],
                "max_scf_steps": 12,
            }
        )
    )

    payload = adapter.write_weak_scaling_report(tmp_path / "weak.json", [run0, run1])

    assert payload["num_runs"] == 2
    assert payload["baseline_max_block_wall_time_seconds"] == 10.0
    assert payload["num_blocks"] == 2
    assert payload["success_rate"] == 1.0
    assert payload["max_block_wall_time"] == 12.5
    assert payload["mean_block_wall_time"] == 11.0
    assert payload["solver_used"] == ["NTPOLY"]
    assert payload["ntpoly_methods"] == [2]
    assert payload["runs"][0]["weak_scaling_efficiency_vs_baseline"] == 1.0
    assert payload["runs"][1]["weak_scaling_efficiency_vs_baseline"] == 0.8
    assert payload["runs"][1]["scheduled_ranks"] == []
    assert payload["runs"][1]["ranks_with_results"] == []
    assert payload["runs"][1]["success_rate"] == 1.0
    assert payload["runs"][1]["solver_used"] == ["NTPOLY"]
    assert payload["runs"][1]["ntpoly_methods"] == [2]
    assert payload["runs"][1]["max_scf_steps"] == 12
    assert json.loads((tmp_path / "weak.json").read_text()) == payload


def test_weak_scaling_report_cli(tmp_path):
    repo_root = Path(__file__).resolve().parents[4]
    run0 = tmp_path / "run0"
    run0.mkdir()
    (run0 / "run_summary.json").write_text(
        json.dumps(
            {
                "workdir": str(run0),
                "num_blocks": 1,
                "num_scheduled_ranks": 1,
                "num_ranks_with_results": 1,
                "num_machines": 1,
                "num_successful_results": 1,
                "num_failed_results": 0,
                "num_converged_results": 1,
                "max_block_wall_time_seconds": 2.0,
                "mean_block_wall_time_seconds": 2.0,
            }
        )
    )
    output = tmp_path / "report.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "ewf_weak_scaling_report.py"),
            str(run0),
            "--output",
            str(output),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text())
    assert payload["num_runs"] == 1
    assert payload["num_blocks"] == 1
    assert payload["success_rate"] == 1.0
    assert payload["weak_scaling_efficiency_vs_baseline"] == 1.0
    assert payload["runs"][0]["workdir"] == str(run0)
    assert '"num_runs": 1' in completed.stdout


def test_project_results_to_ewf_keeps_only_core_orbital_ownership(tmp_path):
    _write_full_dm(
        tmp_path / "block_0000.DM",
        numh=[1, 2, 2, 2, 1, 1],
        columns=[[1], [2, 3], [2, 3], [4, 5], [4], [6]],
        density_rows=[[[1.0], [2.0, 0.2], [0.2, 3.0], [4.0, 0.4], [0.4], [6.0]]],
    )
    _write_full_hsx(
        tmp_path / "block_0000.HSX",
        numh=[1, 2, 2, 2, 1, 1],
        columns=[[1], [2, 3], [2, 3], [4, 5], [4], [6]],
        hamiltonian_rows=[[[1.0], [2.0, 0.2], [0.2, 3.0], [4.0, 0.4], [0.4], [6.0]]],
        overlap_rows=[[1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0], [1.0]],
    )
    (tmp_path / "blocks.json").write_text(
        json.dumps(
            [
                {
                    "block_id": 0,
                    "machine_id": 2,
                    "core_atom_start": 10,
                    "core_atom_end": 12,
                    "input_atom_start": 9,
                    "input_atom_end": 13,
                    "local_to_global_atom_index": [9, 10, 11, 12],
                }
            ]
        )
    )
    (tmp_path / "results.json").write_text(
        json.dumps(
            [
                {
                    "block_id": 0,
                    "rank": 7,
                    "returncode": 0,
                    "converged": True,
                    "total_energy_ev": -1.5,
                    "density_matrix_path": str(tmp_path / "block_0000.DM"),
                    "hamiltonian_matrix_path": str(tmp_path / "block_0000.HSX"),
                    "overlap_matrix_path": str(tmp_path / "block_0000.HSX"),
                    "orbital_index_path": str(tmp_path / "block_0000.ORB_INDX"),
                    "output_path": str(tmp_path / "siesta.out"),
                    "atom_orbital_ranges": {"9": [0, 1], "10": [1, 3], "11": [3, 5], "12": [5, 6]},
                }
            ]
        )
    )

    projected = adapter.project_results_to_ewf(tmp_path)

    assert len(projected) == 1
    result = projected[0]
    assert result.block_id == 0
    assert result.machine_id == 2
    assert result.rank == 7
    assert result.core_atom_range == (10, 12)
    assert result.input_atom_range == (9, 13)
    assert result.core_atoms == (10, 11)
    assert result.buffer_atoms == (9, 12)
    assert result.core_atom_orbital_ranges == {10: (1, 3), 11: (3, 5)}
    assert result.to_metadata()["core_atom_orbital_ranges"] == {"10": [1, 3], "11": [3, 5]}
    assert result.read_core_density_matrix().nnz == 7
    assert result.read_core_hsx_matrix().nnz == 7

    payload = adapter.write_ewf_results_manifest(tmp_path)

    assert payload[0]["core_atoms"] == [10, 11]
    assert payload[0]["buffer_atoms"] == [9, 12]
    assert payload[0]["core_matrix_metadata"]["hamiltonian_overlap"] == {
        "norbitals": 4,
        "nnz": 7,
        "orbital_start": 1,
        "orbital_end": 5,
    }
    assert payload[0]["core_matrix_metadata"]["density"] == {
        "kind": "density",
        "norbitals": 4,
        "nnz": 7,
        "orbital_start": 1,
        "orbital_end": 5,
    }
    assert json.loads((tmp_path / "ewf_results.json").read_text()) == payload


def test_project_results_to_ewf_rejects_incomplete_orbital_ranges(tmp_path):
    (tmp_path / "blocks.json").write_text(
        json.dumps(
            [
                {
                    "block_id": 0,
                    "core_atom_start": 0,
                    "core_atom_end": 2,
                    "input_atom_start": 0,
                    "input_atom_end": 2,
                    "local_to_global_atom_index": [0, 1],
                }
            ]
        )
    )
    (tmp_path / "results.json").write_text(
        json.dumps(
            [
                {
                    "block_id": 0,
                    "returncode": 0,
                    "converged": True,
                    "density_matrix_path": "x.DM",
                    "hamiltonian_matrix_path": "x.HSX",
                    "overlap_matrix_path": "x.HSX",
                    "orbital_index_path": "x.ORB_INDX",
                    "atom_orbital_ranges": {"0": [0, 1]},
                }
            ]
        )
    )

    with pytest.raises(ValueError, match="missing core orbital ranges"):
        adapter.project_results_to_ewf(tmp_path)


def test_project_results_to_ewf_rejects_matrix_orbital_count_mismatch(tmp_path):
    (tmp_path / "blocks.json").write_text(
        json.dumps(
            [
                {
                    "block_id": 0,
                    "core_atom_start": 0,
                    "core_atom_end": 1,
                    "input_atom_start": 0,
                    "input_atom_end": 1,
                    "local_to_global_atom_index": [0],
                }
            ]
        )
    )
    (tmp_path / "results.json").write_text(
        json.dumps(
            [
                {
                    "block_id": 0,
                    "returncode": 0,
                    "converged": True,
                    "density_matrix_path": "x.DM",
                    "hamiltonian_matrix_path": "x.HSX",
                    "overlap_matrix_path": "x.HSX",
                    "orbital_index_path": "x.ORB_INDX",
                    "atom_orbital_ranges": {"0": [0, 3]},
                    "matrix_metadata": {"density": {"norbitals": 2}},
                }
            ]
        )
    )

    with pytest.raises(ValueError, match="does not match ORB_INDX"):
        adapter.project_results_to_ewf(tmp_path)


def test_project_results_to_ewf_rejects_missing_block_results(tmp_path):
    (tmp_path / "blocks.json").write_text(
        json.dumps(
            [
                {"block_id": 0, "core_atom_start": 0, "core_atom_end": 1, "input_atom_start": 0, "input_atom_end": 1},
                {"block_id": 1, "core_atom_start": 1, "core_atom_end": 2, "input_atom_start": 1, "input_atom_end": 2},
            ]
        )
    )
    (tmp_path / "results.json").write_text(json.dumps([{"block_id": 0, "returncode": 0, "converged": True}]))

    with pytest.raises(ValueError, match="Missing SIESTA results"):
        adapter.project_results_to_ewf(tmp_path, require_matrices=False)


def test_validate_ewf_results_rejects_uncovered_boundary_bonds(tmp_path):
    (tmp_path / "blocks.json").write_text(
        json.dumps(
            [
                {
                    "block_id": 0,
                    "core_atom_start": 0,
                    "core_atom_end": 1,
                    "input_atom_start": 0,
                    "input_atom_end": 1,
                    "local_to_global_atom_index": [0],
                }
            ]
        )
    )
    (tmp_path / "results.json").write_text(
        json.dumps(
            [
                {
                    "block_id": 0,
                    "returncode": 0,
                    "converged": True,
                    "atom_orbital_ranges": {"0": [0, 1]},
                }
            ]
        )
    )
    (tmp_path / "boundary.json").write_text(
        json.dumps(
            {
                "uncovered_boundary_bonds": [
                    {
                        "block_id": 0,
                        "atom_i": 0,
                        "atom_j": 1,
                    }
                ]
            }
        )
    )
    (tmp_path / "embedding_contract.json").write_text(
        json.dumps(
            {
                "num_pending_embedding_terms": 1,
                "terms": [
                    {
                        "block_id": 0,
                        "bond_atoms": [0, 1],
                        "status": "invalid_uncovered_boundary",
                    }
                ],
            }
        )
    )
    (tmp_path / "electron_constraint.json").write_text(
        json.dumps({"electron_count_deviation": -0.5})
    )
    (tmp_path / "boundary_corrections.json").write_text(
        json.dumps({"num_unparameterized_corrections": 1})
    )

    report = adapter.validate_ewf_results(tmp_path, natoms=1, require_matrices=False)

    assert report.ok is False
    assert "boundary bond 0-1 is not covered" in report.errors[0]
    assert "embedding term [0, 1] has uncovered boundary" in report.errors[1]
    assert "1 boundary embedding terms require" in report.warnings[0]
    assert "electron-count deviation -0.5" in report.warnings[1]
    assert "1 boundary correction slots are not parameterized" in report.warnings[2]


def test_validate_ewf_results_rejects_non_ntpoly_solver_metadata():
    result = adapter.SiestaEwfResult(
        block_id=0,
        machine_id=0,
        rank=0,
        core_atom_range=(0, 1),
        input_atom_range=(0, 1),
        core_atoms=(0,),
        buffer_atoms=(),
        core_atom_orbital_ranges={0: (0, 1)},
        converged=True,
        total_energy_ev=-1.0,
        density_matrix_path=Path("x.DM"),
        hamiltonian_matrix_path=Path("x.HSX"),
        overlap_matrix_path=Path("x.HSX"),
        orbital_index_path=Path("x.ORB_INDX"),
        output_path=Path("siesta.out"),
        matrix_metadata={"elsi": {"solver_used": ["ELPA"]}},
    )

    report = adapter.validate_ewf_results([result], natoms=1, require_matrices=False)

    assert report.ok is False
    assert "expected ['NTPOLY']" in report.errors[0]


def test_validate_ewf_results_rejects_non_trs2_ntpoly_method():
    result = adapter.SiestaEwfResult(
        block_id=0,
        machine_id=0,
        rank=0,
        core_atom_range=(0, 1),
        input_atom_range=(0, 1),
        core_atoms=(0,),
        buffer_atoms=(),
        core_atom_orbital_ranges={0: (0, 1)},
        converged=True,
        total_energy_ev=-1.0,
        density_matrix_path=Path("x.DM"),
        hamiltonian_matrix_path=Path("x.HSX"),
        overlap_matrix_path=Path("x.HSX"),
        orbital_index_path=Path("x.ORB_INDX"),
        output_path=Path("siesta.out"),
        matrix_metadata={"elsi": {"solver_used": ["NTPOLY"], "last_solver_settings": {"nt_method": 1}}},
    )

    report = adapter.validate_ewf_results([result], natoms=1, require_matrices=False)

    assert report.ok is False
    assert "expected TRS2 method 2" in report.errors[0]


def test_assemble_global_matrices_from_core_owned_blocks(tmp_path):
    block0_dm = tmp_path / "block_0000.DM"
    block0_hsx = tmp_path / "block_0000.HSX"
    block1_dm = tmp_path / "block_0001.DM"
    block1_hsx = tmp_path / "block_0001.HSX"
    _write_full_dm(block0_dm, [2, 2], [[1, 2], [1, 2]], [[[1.0, 0.1], [0.1, 2.0]]])
    _write_full_hsx(block0_hsx, [2, 2], [[1, 2], [1, 2]], [[[10.0, 1.0], [1.0, 20.0]]], [[1.0, 0.0], [0.0, 1.0]])
    _write_full_dm(block1_dm, [1], [[1]], [[[3.0]]])
    _write_full_hsx(block1_hsx, [1], [[1]], [[[30.0]]], [[1.0]])
    results = [
        adapter.SiestaEwfResult(
            block_id=0,
            machine_id=0,
            rank=0,
            core_atom_range=(0, 1),
            input_atom_range=(0, 1),
            core_atoms=(0,),
            buffer_atoms=(),
            core_atom_orbital_ranges={0: (0, 2)},
            converged=True,
            total_energy_ev=-1.0,
            density_matrix_path=block0_dm,
            hamiltonian_matrix_path=block0_hsx,
            overlap_matrix_path=block0_hsx,
            orbital_index_path=None,
            output_path=None,
        ),
        adapter.SiestaEwfResult(
            block_id=1,
            machine_id=0,
            rank=1,
            core_atom_range=(1, 2),
            input_atom_range=(1, 2),
            core_atoms=(1,),
            buffer_atoms=(),
            core_atom_orbital_ranges={1: (0, 1)},
            converged=True,
            total_energy_ev=-2.0,
            density_matrix_path=block1_dm,
            hamiltonian_matrix_path=block1_hsx,
            overlap_matrix_path=block1_hsx,
            orbital_index_path=None,
            output_path=None,
        ),
    ]

    global_mats = adapter.assemble_global_matrices(results, natoms=2)

    assert global_mats.atom_orbital_ranges == {0: (0, 2), 1: (2, 3)}
    assert global_mats.rows.tolist() == [0, 0, 1, 1, 2]
    assert global_mats.cols.tolist() == [0, 1, 0, 1, 2]
    assert global_mats.block_ids.tolist() == [0, 0, 0, 0, 1]
    assert global_mats.density.shape == (1, 5)
    assert global_mats.density[0].tolist() == [1.0, 0.1, 0.1, 2.0, 3.0]
    assert global_mats.hamiltonian[0].tolist() == [10.0, 1.0, 1.0, 20.0, 30.0]
    assert global_mats.overlap.tolist() == [1.0, 0.0, 0.0, 1.0, 1.0]
    assert global_mats.to_metadata()["norbitals"] == 3
    assert global_mats.density_overlap_trace_by_spin == (6.0,)
    assert global_mats.density_overlap_trace_total == 6.0
    assert global_mats.to_metadata()["density_overlap_trace_total"] == 6.0


def test_write_global_matrices_manifest_from_workdir(tmp_path):
    _write_full_dm(tmp_path / "block_0000.DM", [1], [[1]], [[[1.0]]])
    _write_full_hsx(tmp_path / "block_0000.HSX", [1], [[1]], [[[10.0]]], [[1.0]])
    (tmp_path / "blocks.json").write_text(
        json.dumps(
            [
                {
                    "block_id": 0,
                    "core_atom_start": 0,
                    "core_atom_end": 1,
                    "input_atom_start": 0,
                    "input_atom_end": 1,
                    "local_to_global_atom_index": [0],
                }
            ]
        )
    )
    (tmp_path / "results.json").write_text(
        json.dumps(
            [
                {
                    "block_id": 0,
                    "returncode": 0,
                    "converged": True,
                    "density_matrix_path": str(tmp_path / "block_0000.DM"),
                    "hamiltonian_matrix_path": str(tmp_path / "block_0000.HSX"),
                    "overlap_matrix_path": str(tmp_path / "block_0000.HSX"),
                    "orbital_index_path": str(tmp_path / "block_0000.ORB_INDX"),
                    "atom_orbital_ranges": {"0": [0, 1]},
                    "matrix_metadata": {
                        "density": {"norbitals": 1},
                        "hamiltonian_overlap": {"norbitals": 1},
                    },
                }
            ]
        )
    )

    payload = adapter.write_global_matrices_manifest(tmp_path, natoms=1)

    assert payload["natoms"] == 1
    assert payload["norbitals"] == 1
    assert payload["nnz"] == 1
    assert payload["density_overlap_trace_total"] == 1.0
    assert json.loads((tmp_path / "global_matrices.json").read_text()) == payload


def test_validate_ewf_results_reports_core_owned_contract_and_energy_policy(tmp_path):
    dm_path = tmp_path / "block_0000.DM"
    hsx_path = tmp_path / "block_0000.HSX"
    _write_full_dm(dm_path, [1, 1], [[1], [2]], [[[1.0], [2.0]]])
    _write_full_hsx(hsx_path, [1, 1], [[1], [2]], [[[10.0], [20.0]]], [[1.0], [1.0]])
    result = adapter.SiestaEwfResult(
        block_id=0,
        machine_id=0,
        rank=0,
        core_atom_range=(0, 2),
        input_atom_range=(0, 2),
        core_atoms=(0, 1),
        buffer_atoms=(),
        core_atom_orbital_ranges={0: (0, 1), 1: (1, 2)},
        converged=True,
        total_energy_ev=-3.5,
        density_matrix_path=dm_path,
        hamiltonian_matrix_path=hsx_path,
        overlap_matrix_path=hsx_path,
        orbital_index_path=None,
        output_path=None,
    )

    report = adapter.validate_ewf_results([result], natoms=2)

    assert report.ok is True
    assert report.errors == ()
    assert report.nblocks == 1
    assert report.ncore_atoms == 2
    assert report.norbitals == 2
    assert report.nnz == 2
    assert report.density_overlap_trace_total == 3.0
    assert report.density_overlap_trace_by_spin == (3.0,)
    assert report.total_block_energy_ev == -3.5
    assert report.energy_policy == "diagnostic_block_sum_not_embedded_total"
    assert "not an embedded total energy" in " ".join(report.warnings)


def test_validate_ewf_results_warns_when_internal_block_has_insufficient_buffer(tmp_path):
    results = []
    for atom in range(3):
        dm_path = tmp_path / f"block_{atom:04d}.DM"
        hsx_path = tmp_path / f"block_{atom:04d}.HSX"
        _write_full_dm(dm_path, [1], [[1]], [[[float(atom + 1)]]])
        _write_full_hsx(hsx_path, [1], [[1]], [[[float(10 * (atom + 1))]]], [[1.0]])
        results.append(
            adapter.SiestaEwfResult(
                block_id=atom,
                machine_id=0,
                rank=0,
                core_atom_range=(atom, atom + 1),
                input_atom_range=(atom, atom + 1),
                core_atoms=(atom,),
                buffer_atoms=(),
                core_atom_orbital_ranges={atom: (0, 1)},
                converged=True,
                total_energy_ev=None,
                density_matrix_path=dm_path,
                hamiltonian_matrix_path=hsx_path,
                overlap_matrix_path=hsx_path,
                orbital_index_path=None,
                output_path=None,
            )
        )

    report = adapter.validate_ewf_results(results, natoms=3, min_buffer_atoms=1)

    assert report.ok is True
    assert report.errors == ()
    assert any("Block 1 has 0 buffer atoms" in warning for warning in report.warnings)


def test_write_validation_manifest_from_workdir(tmp_path):
    _write_full_dm(tmp_path / "block_0000.DM", [1], [[1]], [[[1.0]]])
    _write_full_hsx(tmp_path / "block_0000.HSX", [1], [[1]], [[[10.0]]], [[1.0]])
    (tmp_path / "blocks.json").write_text(
        json.dumps(
            [
                {
                    "block_id": 0,
                    "core_atom_start": 0,
                    "core_atom_end": 1,
                    "input_atom_start": 0,
                    "input_atom_end": 1,
                    "local_to_global_atom_index": [0],
                }
            ]
        )
    )
    (tmp_path / "results.json").write_text(
        json.dumps(
            [
                {
                    "block_id": 0,
                    "returncode": 0,
                    "converged": True,
                    "total_energy_ev": -1.0,
                    "density_matrix_path": str(tmp_path / "block_0000.DM"),
                    "hamiltonian_matrix_path": str(tmp_path / "block_0000.HSX"),
                    "overlap_matrix_path": str(tmp_path / "block_0000.HSX"),
                    "orbital_index_path": str(tmp_path / "block_0000.ORB_INDX"),
                    "atom_orbital_ranges": {"0": [0, 1]},
                    "matrix_metadata": {
                        "density": {"norbitals": 1},
                        "hamiltonian_overlap": {"norbitals": 1},
                    },
                }
            ]
        )
    )

    payload = adapter.write_validation_manifest(tmp_path, natoms=1)

    assert payload["ok"] is True
    assert payload["nblocks"] == 1
    assert payload["ncore_atoms"] == 1
    assert payload["energy_policy"] == "diagnostic_block_sum_not_embedded_total"
    assert json.loads((tmp_path / "validation.json").read_text()) == payload


def test_assemble_global_matrices_rejects_duplicate_or_missing_core_atoms(tmp_path):
    template = dict(
        machine_id=0,
        rank=0,
        input_atom_range=(0, 1),
        buffer_atoms=(),
        core_atom_orbital_ranges={0: (0, 1)},
        converged=True,
        total_energy_ev=None,
        density_matrix_path=None,
        hamiltonian_matrix_path=None,
        overlap_matrix_path=None,
        orbital_index_path=None,
        output_path=None,
    )
    duplicate = [
        adapter.SiestaEwfResult(block_id=0, core_atom_range=(0, 1), core_atoms=(0,), **template),
        adapter.SiestaEwfResult(block_id=1, core_atom_range=(0, 1), core_atoms=(0,), **template),
    ]
    with pytest.raises(ValueError, match="Duplicate core atom ownership"):
        adapter.assemble_global_matrices(duplicate)

    missing = [adapter.SiestaEwfResult(block_id=0, core_atom_range=(0, 1), core_atoms=(0,), **template)]
    with pytest.raises(ValueError, match="Missing core atom ownership"):
        adapter.assemble_global_matrices(missing, natoms=2)


def test_driver_dry_run_writes_manifests_and_aggregate(tmp_path):
    repo_root = Path(__file__).resolve().parents[4]
    env = os.environ.copy()
    env.update(
        {
            "EWF_WORKDIR": str(tmp_path / "runs"),
            "EWF_NUM_MACHINES": "2",
            "EWF_PROCS_PER_MACHINE": "1",
            "EWF_BLOCK_ATOMS": "200",
            "EWF_BLOCK_BUFFER_ATOMS": "5",
        }
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "ewf_siesta_driver.py"),
            str(repo_root / "testcases" / "0386.fdf"),
            "--pseudo",
            str(repo_root / "testcases" / "C.psf"),
            "--pseudo",
            str(repo_root / "testcases" / "H.psf"),
            "--rank",
            "0",
            "--size",
            "2",
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    workdir = tmp_path / "runs"
    blocks = json.loads((workdir / "blocks.json").read_text())
    assert len(blocks) == 2
    assert (workdir / "result_rank_0000.json").exists()
    assert json.loads((workdir / "results.json").read_text()) == []
    assert (workdir / "weak_scaling_report.json").exists()
    schedule = json.loads((workdir / "schedule.json").read_text())
    assert schedule["num_ranks"] == 2
    assert schedule["total_ranks"] == 2
    assert schedule["block_owner_rank"] == {"0": 0, "1": 1}
    summary = json.loads((workdir / "run_summary.json").read_text())
    assert summary["scheduled_ranks"] == [0, 1]
    assert summary["success_rate"] == 0.0
    assert summary["num_results"] == 0
    assert [block["rank"] for block in summary["blocks"]] == [0, 1]


def test_driver_dry_run_supports_group_partitioning(tmp_path):
    repo_root = Path(__file__).resolve().parents[4]
    env = os.environ.copy()
    env.update(
        {
            "EWF_WORKDIR": str(tmp_path / "runs"),
            "EWF_NUM_MACHINES": "2",
            "EWF_PROCS_PER_MACHINE": "1",
            "EWF_GROUP_SIZE_ATOMS": "6",
            "EWF_BLOCK_GROUPS": "10",
            "EWF_BLOCK_BUFFER_GROUPS": "1",
        }
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "ewf_siesta_driver.py"),
            str(repo_root / "testcases" / "0386.fdf"),
            "--rank",
            "0",
            "--size",
            "2",
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    blocks = json.loads((tmp_path / "runs" / "blocks.json").read_text())
    assert len(blocks) == 7
    assert blocks[0]["core_atom_start"] == 0
    assert blocks[0]["core_atom_end"] == 60
    assert blocks[1]["core_atom_start"] == 60
    assert blocks[1]["input_atom_start"] == 54


def test_gen_py_orders_atoms_by_polyethylene_chain_groups(tmp_path):
    repo_root = Path(__file__).resolve().parents[4]

    completed = subprocess.run(
        [sys.executable, str(repo_root / "testcases" / "gen.py"), "4"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )

    fdf_path = tmp_path / "pe4.fdf"
    fdf_path.write_text(completed.stdout)
    fdf = adapter.parse_fdf(fdf_path)

    assert [atom.species for atom in fdf.atoms[:14]] == [1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2]


def test_prepare_siesta_workflow_runs_dry_rank_and_finalize(tmp_path):
    repo_root = Path(__file__).resolve().parents[4]
    config = adapter.SiestaRunConfig(
        num_machines=2,
        procs_per_machine=1,
        threads_per_proc=1,
        workdir=tmp_path / "runs",
        siesta_bin=None,
        block_atoms=200,
        buffer_atoms=5,
        block_groups=None,
        group_size_atoms=None,
        buffer_groups=0,
        terminal_cap_atoms=0,
        dry_run=True,
    )

    workflow = adapter.prepare_siesta_workflow(
        repo_root / "testcases" / "0386.fdf",
        config=config,
        pseudopotentials=[repo_root / "testcases" / "C.psf"],
    )

    assert workflow.natoms == 386
    assert len(workflow.blocks) == 2
    workflow.write_inputs()
    parsed, completed = workflow.run_rank(rank=0, machine_id=0, local_rank=0)
    payload = workflow.finalize()

    assert parsed == []
    assert completed == []
    assert (config.workdir / "blocks.json").exists()
    assert (config.workdir / "schedule.json").exists()
    assert (config.workdir / "boundary.json").exists()
    assert (config.workdir / "embedding_contract.json").exists()
    assert (config.workdir / "boundary_corrections.json").exists()
    assert (config.workdir / "result_rank_0000.json").exists()
    assert payload["results"] == []
    assert payload["run_summary"]["scheduled_ranks"] == [0, 1]
    assert payload["weak_scaling_report"]["scheduled_ranks"] == [0, 1]
    assert payload["validation"] is None


def test_driver_fails_when_mpi_size_is_smaller_than_configured_workers(tmp_path):
    repo_root = Path(__file__).resolve().parents[4]
    env = os.environ.copy()
    env.update(
        {
            "EWF_WORKDIR": str(tmp_path / "runs"),
            "EWF_NUM_MACHINES": "2",
            "EWF_PROCS_PER_MACHINE": "2",
        }
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "ewf_siesta_driver.py"),
            str(repo_root / "testcases" / "0386.fdf"),
            "--rank",
            "0",
            "--size",
            "3",
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "smaller than configured workers" in completed.stderr


def test_driver_failed_siesta_run_writes_validation_without_traceback(tmp_path):
    repo_root = Path(__file__).resolve().parents[4]
    fdf_path = tmp_path / "input.fdf"
    fdf_path.write_text(
        "\n".join(
            [
                "SystemLabel failcase",
                "NumberOfAtoms 1",
                "NumberOfSpecies 1",
                "%block AtomicCoordinatesAndAtomicSpecies",
                "0.0 0.0 0.0 1",
                "%endblock AtomicCoordinatesAndAtomicSpecies",
            ]
        )
        + "\n"
    )
    runner = tmp_path / "fail-siesta.py"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "print('SCF_NOT_CONV: SCF did not converge in maximum number of steps (required).')",
                "raise SystemExit(1)",
            ]
        )
        + "\n"
    )
    runner.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "EWF_WORKDIR": str(tmp_path / "runs"),
            "EWF_NUM_MACHINES": "1",
            "EWF_PROCS_PER_MACHINE": "1",
            "EWF_BLOCK_ATOMS": "1",
            "EWF_SIESTA_BIN": str(runner),
            "EWF_SIESTA_DRY_RUN": "false",
        }
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "ewf_siesta_driver.py"),
            str(fdf_path),
            "--rank",
            "0",
            "--size",
            "1",
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "validation_error:" in completed.stderr
    assert "Traceback" not in completed.stderr
    validation = json.loads((tmp_path / "runs" / "validation.json").read_text())
    assert validation["ok"] is False
    assert "failed with return code 1" in validation["errors"][0]
