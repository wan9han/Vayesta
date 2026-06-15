# SIESTA Adapter Interface

This package is the first integration layer for the EWF -> SIESTA -> EWF route.
It is deliberately independent of the PySCF-backed EWF internals so the SIESTA
block workflow can be developed and tested with existing `.fdf` inputs.

The current coarse block calculation is an engineering backend contract, not a
complete physical EWF approximation by itself.  Running each block as an
isolated SIESTA system validates scheduling, local input generation, matrix-file
production, and result mapping.  Global physical meaning requires the EWF layer
to define the embedding environment, core-only projection, electron/chemical
potential constraints, boundary treatment, and double-counting corrections.
Until that layer is wired in, block energies and density matrices should not be
summed as a final global result.

## Current Data Contract

`SiestaBlock` stores the block ownership and global mapping:

- `core_atom_start`, `core_atom_end`: half-open global atom range owned by the block.
- `input_atom_start`, `input_atom_end`: half-open global atom range sent to SIESTA, including buffer atoms.
- `machine_id`: simulated machine assignment from `block_id % EWF_NUM_MACHINES`.
- `local_to_global_atom_index`: written to `block.json` for result projection back to the global system.

`add_siesta_block_fragments(fragmentation, blocks)` bridges the same block
definition into Vayesta's native fragmentation API.  It calls
`fragmentation.add_atomic_fragment(...)` with the core atoms only and stores the
SIESTA input/buffer metadata on the returned fragment:

- `siesta_block_id`
- `siesta_machine_id`
- `siesta_core_atoms`
- `siesta_input_atoms`
- `siesta_buffer_atoms`
- `siesta_core_atom_range`
- `siesta_input_atom_range`

After a SIESTA run has been projected with `project_results_to_ewf`,
`attach_siesta_results_to_fragments(fragments, results)` maps those
`SiestaEwfResult` objects back to the same fragments by `siesta_block_id`.  It
attaches paths and core matrix metadata as fragment attributes, including
`siesta_ewf_result`, `siesta_density_matrix_path`,
`siesta_hamiltonian_matrix_path`, `siesta_core_atom_orbital_ranges`, and
`siesta_core_matrix_metadata`.  `load_siesta_results_to_fragments(workdir,
fragments)` combines projection from a run directory with this attachment step.

Example:

```python
from vayesta.siesta import add_siesta_block_fragments, partition_contiguous_atoms

blocks = partition_contiguous_atoms(natoms, block_atoms=1000, buffer_atoms=100, num_machines=4)
with emb.sao_fragmentation() as frag:
    add_siesta_block_fragments(frag, blocks)
```

`generate_block_directories` writes:

- `input.fdf`: local SIESTA input with only input atoms.
- `block.json`: global/local atom mapping and buffer metadata.
- `run.sh`: standalone SIESTA invocation with BLAS thread environment.
- pseudopotential copies such as `C.psf` and `H.psf`.

For direct Python integration, `prepare_siesta_workflow(...)` returns a
`SiestaBlockWorkflow` object which owns the complete adapter workflow:

```python
from vayesta.siesta import prepare_siesta_workflow, read_run_config

config = read_run_config()
workflow = prepare_siesta_workflow("testcases/0386.fdf", config=config, pseudopotentials=["testcases/C.psf"])
workflow.write_inputs()
parsed_results, completed = workflow.run_rank(rank=0, machine_id=0, local_rank=0)
finalized = workflow.finalize()
```

The command-line driver is a thin MPI/barrier wrapper around this object.  This
keeps the EWF-facing Python API and the script path on the same implementation.

The workflow also writes `boundary.json`.  This manifest infers covalent bonds
from the FDF coordinates and species labels, lists bonds crossing each block's
core boundary, and records whether the other end of each boundary bond is
covered by the SIESTA input/buffer atoms.  `validation.json` treats uncovered
boundary bonds as errors.

It also writes `embedding_contract.json`, a conservative boundary contract
derived from `boundary.json`.  Each covered core-boundary bond becomes a pending
embedding term requiring an embedding potential and an energy correction.  The
adapter records these required corrections but does not apply them yet.

The workflow then writes `boundary_corrections.json`.  This is a structured
placeholder plan derived from `embedding_contract.json`: every pending boundary
embedding term receives a correction slot with null embedding potential, energy
correction, and electron-count correction fields.  `validation.json` warns about
these unparameterized slots so downstream consumers cannot silently interpret
the diagnostic block calculation as a complete embedded result.

For the same reason, real runs also write `physical_readiness.json`.  This
manifest separates engineering readiness from physical readiness:
`backend_artifacts_ready=true` means the SIESTA block outputs passed the current
adapter checks, while `embedded_observable_ready=false` means the result still
lacks embedding potentials, boundary corrections, or an applied electron-number
constraint.

The generated `input.fdf` forces the intended SIESTA/ELSI solver path and the
output files needed by the first reader contract:

- `SolutionMethod ELSI`
- `ELSI.Solver ntpoly`
- `ELSI.NTPoly.Method 2`
- `ELSI.NTPoly.Filter 1.0e-9`
- `ELSI.NTPoly.Tolerance 1.0e-6`
- `MaxSCFIterations 150`
- `DM.NumberPulay 6`
- `DM.MixingWeight 0.050000`
- `WriteDM true`
- `SaveHS true`
- `WriteOrbitalIndex true`

This keeps each local SIESTA block on the ELSI -> NTPoly density-matrix
purification path instead of the ELSI default ELPA diagonalization path.  In the
bundled NTPoly source, `TRS2` is the density-matrix purification routine; the
adapter records the requested NTPoly method in the generated input and validates
the actually used ELSI solver from `elsi_log.json` during output parsing and
`validation.json` generation.

`read_siesta_output` currently returns the scalar text-output contract:

- `block_id`
- `converged`
- `total_energy_ev`
- `wall_time_seconds`
- `density_matrix_path`
- `hamiltonian_matrix_path`
- `overlap_matrix_path`
- `orbital_index_path`
- `atom_orbital_ranges`: global atom index to zero-based half-open orbital range.
- `output_path`
- `matrix_metadata`: header-level DM/HSX metadata, including orbital count, spin count, supercell, HSX precision, atom count, and species count where available.

The matrix readers expose sparse COO-style arrays with zero-based orbital
indices:

- `read_density_matrix_sparse(path)`: returns `SiestaDensityMatrix` with `rows`, `cols`, and `density[nspin, nnz]`.
- `read_hsx_sparse(path)`: returns `SiestaHsxMatrix` with `rows`, `cols`, `hamiltonian[nspin, nnz]`, and `overlap[nnz]`.

Both matrix containers can produce a core-orbital subblock summary from the
`core_atom_orbital_ranges` map.  `ewf_results.json` stores this compact
`core_matrix_metadata` summary; the full sparse values remain in the SIESTA
matrix files and are read on demand.

`SiestaEwfResult` also provides:

- `read_core_density_matrix()`
- `read_core_hsx_matrix()`

These return sparse core-orbital subblocks using the paths and
`core_atom_orbital_ranges` already carried by the result object.

For a physical EWF collection step, the consumer should treat these outputs as
local backend artifacts.  The core range in `block.json` defines ownership, while
the buffer range is environmental context.  Boundary and buffer contributions
must be projected or corrected by the EWF layer before constructing a global
observable.

SIESTA writes the density matrix as `<SystemLabel>.DM` when `WriteDM true`.
It writes the orbital map as `<SystemLabel>.ORB_INDX` by default.  Hamiltonian
and overlap discovery currently looks for `*.HSX`; if a run does not produce
HSX output, enable the SIESTA `SaveHS` / `Write.HS` path in the input.

## Rank Scheduling

The adapter uses the topology from `codex.md`:

```text
machine_id = rank / EWF_PROCS_PER_MACHINE
local_rank = rank % EWF_PROCS_PER_MACHINE
```

Blocks are first assigned to simulated machines with
`block.machine_id = block_id % EWF_NUM_MACHINES`.  Within each machine, the
machine's blocks are round-robin assigned to `local_rank`, so every rank can run
one or more local SIESTA tasks instead of leaving all work on `local_rank=0`.

The outer MPI ranks are EWF/Vayesta task runners.  They do not cooperate on one
SIESTA block.  In the current adapter contract, one outer rank runs one assigned
block at a time by launching an independent SIESTA subprocess.  Therefore the
active CPU budget is:

```text
active SIESTA block subprocesses * EWF_THREADS_PER_PROC
```

not:

```text
blocks * ranks-per-block * threads
```

When launched under MPI, rank 0 generates the block directories and writes
`blocks.json`, then all ranks enter the scheduling barrier.  Each rank writes
its own `result_rank_XXXX.json`.  After the execution barrier, rank 0 combines
those files into `results.json`, sorted by `block_id`, for the EWF collection
step.

Each local SIESTA run is launched as its own subprocess.  The runner strips
parent MPI launcher variables such as `OMPI_*`, `PMI_*`, and `PMIX_*` from that
subprocess environment so an MPI-enabled SIESTA binary can initialize as an
independent singleton task under an outer EWF `mpirun`.

On the current 28-logical-CPU development machine, the 386-atom polyethylene
diagnostic converged with two large contiguous blocks using:

```text
EWF_NUM_MACHINES=2
EWF_PROCS_PER_MACHINE=1
EWF_THREADS_PER_PROC=2
EWF_GROUP_SIZE_ATOMS=6
EWF_BLOCK_GROUPS=32
EWF_BLOCK_BUFFER_GROUPS=2
EWF_TERMINAL_CAP_ATOMS=2
mpirun -np 2
```

Both blocks used `solver_used=["NTPOLY"]`, `nt_method=2`, and converged in 13
SCF steps.  This is the recommended local stress-test configuration before
trying finer weak-scaling partitions on this workstation.

## Minimal Embedded Closure

The adapter writes a minimal physical closure layer on top of validated SIESTA
block artifacts:

- `boundary_corrections.json` uses the
  `core-owned-buffer-saturated-zero-shift` model.  Boundary atoms are already
  present in the local SIESTA input; the explicit correction term is therefore a
  parameterized zero Hamiltonian shift and zero boundary energy correction until
  a higher EWF layer replaces it.
- `electron_constraint.json` applies a global trace-shift closure so the
  corrected electron count matches the target valence count.
- `embedded_observables.json` combines the block energy sum, boundary energy
  corrections, and corrected electron-count metadata.
- `embedding_benchmark.json` optionally compares the closed embedded observable
  against a full-system or higher-accuracy reference observable.
- `calibrate_boundary_corrections_to_reference(...)` can distribute a known
  reference energy difference over boundary correction slots.  This is a
  calibration path, not a predictive self-consistent boundary potential.
- `predictive_ewf_closure.json` records an unreferenced mean-field EWF closure
  diagnostic: boundary-density SVD bath ranks, external embedding-potential
  expectation values, and a double-counting correction diagnostic.
- `cluster_hamiltonians.json` records solver-ready one-electron cluster
  artifacts.  Each block writes a compressed NPZ with core AO + SVD bath basis
  coefficients, Lowdin orthogonalizer, and orthogonalized H/S/DM arrays.
- `cluster_solver_results.json` consumes those NPZ files with a one-electron
  Lowdin reference solver.  This verifies the solver interface but is not a
  correlated EWF solver.
- `siesta_ao_ordering.json` parses each block `ORB_INDX` into structured AO
  records: block-local orbital index, local/global atom index, species label,
  `n/l/m/zeta`, polarization flag, symbolic orbital name, cutoff radius, and a
  stable label/fingerprint.  This is the handoff point for an external
  SIESTA-to-PySCF AO mapping generator.
- `ao_eri_contract.json` tells an external ERI producer exactly which
  `ao_eri_block_XXXX` arrays to provide, their required block-local AO shapes,
  and the `ao_ordering_fingerprint_block_XXXX` values that will be verified.
  When `siesta_ao_ordering.json` is present, each contract block also embeds
  `siesta_ao_records` and the corresponding AO-ordering fingerprint, so the
  producer can construct a mapping from actual AO identities rather than only
  from array dimensions.
  `write_pyscf_ao_eri_from_contract()` is a small PySCF `int2e` contract
  producer for smoke tests and mapped small systems; each contract block must
  provide explicit `pyscf_ao_indices`, because SIESTA-to-PySCF AO ordering
  cannot be inferred safely.  `apply_pyscf_ao_mapping_to_contract()` injects a
  separate `pyscf_ao_mapping.json` into the contract after validating block
  dimensions and ordering fingerprints.
- `pyscf_external_eri_workflow.json` is written by
  `run_pyscf_external_eri_workflow()`.  It chains the explicit AO mapping,
  PySCF `int2e` producer, AO-to-cluster transform, effective correlated
  second-order solver, `embedded_observables.json`, and
  `physical_readiness.json`.  This gives a single auditable manifest for the
  verified external ERI smoke path; it still requires an explicit AO mapping
  and does not make the second-order prototype a production correlated EWF
  kernel.  A production correlated solver override is available only when
  explicitly requested; it is not the default path.
- `production_correlated_results.json` is written only when the explicit
  production override is enabled.  It does not replace
  `effective_correlated_results.json`; both manifests are kept so downstream
  code can distinguish second-order and production correlated corrections.
- `cluster_two_electron_integrals.json` can be generated from an external
  AO-basis ERI tensor with the same orbital ordering as the SIESTA-returned
  matrices.  It writes per-block `cluster_two_electron_integrals_block_XXXX.npz`
  files and updates `cluster_hamiltonians.json` with
  `two_electron_integrals_npz_path`.  The input NPZ may declare
  `energy_unit="hartree"` or `"ev"`; output `ovov` couplings are always eV.
  Multi-block inputs can provide `ao_eri_block_XXXX` arrays so each local
  SIESTA block can use its own AO dimension.  Each output block records a
  block-local SIESTA AO-ordering fingerprint.  If the input NPZ provides
  `ao_ordering_fingerprint_block_XXXX`, it must match the computed fingerprint;
  otherwise that block is rejected.  Missing fingerprints are allowed but
  marked `unverified_external_ordering`.
- `write_pyscf_ao_mapping_from_siesta_labels()` can generate a conservative
  `pyscf_ao_mapping.json` when the contract contains `siesta_ao_records` and
  the PySCF molecule has matching AO labels.  It matches by atom index, species,
  angular momentum, magnetic component, and occurrence order.  Basis mismatches
  are reported as blockers rather than guessed.
- `effective_correlated_results.json` applies a second-order cluster
  correlation prototype.  By default it uses a configurable Hubbard-like model
  interaction and explicitly records that no ab-initio two-electron integrals
  are used.  If a `cluster_hamiltonians.json` block includes
  `two_electron_integrals_npz_path`, the solver instead consumes that NPZ's
  `ovov` tensor in the cluster eigenbasis and records
  `uses_ab_initio_two_electron_integrals=true`.  The manifest carries
  `ao_ordering_verified`, and `embedded_observables.json` uses that flag to
  label the effective energy policy as verified external ERI, unverified
  external ERI, or model-U correction.
- `effective_interaction_benchmark_scan.json` scans effective-interaction
  strengths against a reference observable and records whether a real
  nonnegative model U can improve or fit the reference energy.

When these files are present and validation passes, `physical_readiness.json`
reports `embedded_observable_ready`.  This is a minimal embedding closure, not a
self-consistent high-level correlated EWF solver.

The public collection helpers are:

- `collect_rank_results(workdir)`: read all `result_rank_XXXX.json` files and return block-ordered result metadata.
- `write_results_manifest(workdir)`: write the block-ordered `results.json` manifest and return the same payload.
- `build_schedule(blocks, num_machines, procs_per_machine)`: build the planned block-to-rank assignment.
- `write_schedule_manifest(workdir, blocks, num_machines, procs_per_machine)`: write that assignment to `schedule.json`.
- `infer_bonds(fdf)`: infer covalent bonds from coordinates and species labels.
- `analyze_block_boundaries(fdf, blocks)`: report core-boundary bonds and buffer coverage.
- `write_boundary_manifest(workdir, fdf, blocks)`: write those diagnostics to `boundary.json`.
- `build_embedding_contract(boundary_payload)`: derive pending embedding/boundary correction terms.
- `write_embedding_contract_manifest(workdir)`: write those terms to `embedding_contract.json`.
- `build_boundary_correction_plan(embedding_contract)`: derive explicit boundary correction slots.
- `write_boundary_corrections_manifest(workdir)`: write those slots to `boundary_corrections.json`.
- `build_physical_readiness_report(workdir)`: report backend readiness versus final embedded-observable readiness.
- `write_physical_readiness_manifest(workdir)`: write that report to `physical_readiness.json`.
- `project_results_to_ewf(workdir)`: combine `blocks.json` and `results.json` into `SiestaEwfResult` objects.
- `attach_siesta_results_to_fragments(fragments, results)`: attach projected SIESTA results to existing Vayesta fragments by `siesta_block_id`.
- `load_siesta_results_to_fragments(workdir, fragments)`: project a run directory and attach its results to fragments.
- `write_ewf_results_manifest(workdir)`: write `ewf_results.json` with core-owned `SiestaEwfResult` metadata.
- `assemble_global_matrices(workdir_or_results, natoms=None)`: assemble core-owned sparse DM/H/S entries into compact global orbital numbering.
- `write_global_matrices_manifest(workdir, natoms=None)`: write `global_matrices.json` with the assembled matrix summary.
- `build_matrix_shape_report(workdir)`: build per-rank/block local matrix M/N/K diagnostics.
- `write_matrix_shape_report_manifest(workdir)`: write `matrix_shape_report.json` with per-process local/core matrix shapes.
- `write_embedded_observables_manifest(workdir)`: write the minimal closed observable manifest.
- `write_embedding_benchmark_manifest(workdir, reference_observables)`: compare embedded observables with a reference.
- `write_embedding_benchmark_from_reference_workdir(workdir, reference_workdir)`: compare embedded observables with another run directory.
- `calibrate_boundary_corrections_to_reference(workdir, reference_total_energy_ev)`: fit boundary energy corrections to a reference total energy.
- `build_predictive_boundary_potential(workdir)`: derive non-reference boundary potentials from returned SIESTA DM/HSX coupling terms.
- `write_predictive_boundary_potential_manifest(workdir)`: write `predictive_embedding_potential.json`.
- `write_predictive_boundary_corrections_manifest(workdir)`: replace `boundary_corrections.json` with predictive boundary-coupling corrections.
- `write_predictive_ewf_closure_manifest(workdir)`: write bath-rank and mean-field double-counting diagnostics from returned SIESTA matrices.
- `write_cluster_hamiltonians_manifest(workdir)`: write per-block cluster Hamiltonian NPZ files and `cluster_hamiltonians.json`.
- `write_cluster_solver_results_manifest(workdir)`: solve the cluster Hamiltonian NPZ files with the one-electron reference solver.
- `write_siesta_ao_ordering_manifest(workdir)`: parse block `ORB_INDX` files into structured AO labels and fingerprints.
- `write_ao_eri_contract_manifest(workdir, energy_unit="ev")`: write the per-block AO ERI producer contract and expected ordering fingerprints.
- `write_pyscf_ao_mapping_from_siesta_labels(mol, contract_path, output_path)`: generate a conservative PySCF AO mapping from contract `siesta_ao_records`.
- `apply_pyscf_ao_mapping_to_contract(contract_path, mapping_path, output_path=None)`: validate and inject explicit PySCF AO indices into the contract.
- `write_pyscf_ao_eri_from_contract(mol, contract_path, output_path)`: write a PySCF `int2e` AO ERI NPZ following the contract when explicit `pyscf_ao_indices` are supplied.
- `run_pyscf_external_eri_workflow(workdir, mol, mapping_path)`: run the explicit-mapping PySCF ERI smoke workflow through effective observables and readiness manifests.
- `write_cluster_two_electron_integrals_from_ao_manifest(workdir, ao_integrals_npz_path, energy_unit=None)`: transform an external AO ERI tensor into per-block cluster eigenbasis `ovov` pair-coupling tensors.
- `write_effective_correlated_results_manifest(workdir)`: compute the second-order effective-interaction correlation correction, using external `ovov` tensors when present and model U otherwise.
- `write_effective_interaction_benchmark_scan_manifest(workdir, reference_observables, u_values_ev)`: quantify effective-interaction model response against a reference.
- `summarize_run(workdir)`: build rank/block success, timing, and matrix-size metrics.
- `write_run_summary_manifest(workdir)`: write those metrics to `run_summary.json`.
- `compare_weak_scaling_runs(workdirs)`: compare multiple `run_summary.json` files.
- `write_weak_scaling_report(output_path, workdirs)`: write a multi-run weak-scaling report.

Set `EWF_PREDICTIVE_BOUNDARY=true` to have `SiestaBlockWorkflow.finalize()`
write the predictive boundary potential and predictive boundary-coupling
corrections automatically.  Set `EWF_PREDICTIVE_BOUNDARY_RERUN=true` as well
to run a second SIESTA pass with per-block `EWF.Embedding.PotentialFile`
inputs.  The current rerun potential is a sparse nonlocal boundary shift
derived from returned SIESTA DM/HSX coupling terms.  It is still a minimal
boundary model, not a full production correlated embedding potential.

If `EWF_AO_ERI_NPZ=/path/to/ao_eri.npz` is set, `finalize()` also runs
`write_cluster_two_electron_integrals_from_ao_manifest()` after cluster
Hamiltonian construction.  `EWF_AO_ERI_ENERGY_UNIT` may be `ev` or `hartree`.
Successful conversion causes the subsequent effective-correlated step to use
external `ovov` tensors instead of the model U coupling.  For weak-scaling runs
with different local block sizes, use `ao_eri_block_0000`,
`ao_eri_block_0001`, ... in the input NPZ.  Add matching
`ao_ordering_fingerprint_block_0000`, `ao_ordering_fingerprint_block_0001`, ...
entries when the ERI producer can prove it used the same block-local SIESTA AO
order.

Set `EWF_PYSCF_AO_MAPPING_MODE=auto` to have `finalize()` try the PySCF ERI
producer path automatically.  This requires `EWF_PYSCF_BASIS`; optional
`EWF_PYSCF_CHARGE`, `EWF_PYSCF_SPIN`, and `EWF_PYSCF_COORD_UNIT` control the
PySCF molecule built from the FDF coordinates.  The auto path writes
`pyscf_ao_mapping.json` and `pyscf_external_eri_workflow.json`.  If PySCF AO
labels do not match the SIESTA `ORB_INDX` records, the workflow records a
blocker instead of guessing a mapping.
Set `EWF_PRODUCTION_CORRELATED_SOLVER=mp2` or `ccsd` only when you explicitly
want the PySCF production solver to replace the second-order external-ERI
correction in `embedded_observables.json`.  The default is `off`, which
preserves the second-order manifest semantics.  When enabled, the workflow
keeps both `effective_correlated_results.json` and
`production_correlated_results.json`, and `embedded_observables.json` records
which one was selected.

`SiestaEwfResult` is the first EWF-facing contract.  It keeps the local SIESTA
matrix file paths and scalar status, but only assigns ownership to core atoms:

- `core_atoms`
- `buffer_atoms`
- `core_atom_orbital_ranges`
- `core_matrix_metadata`
- `core_atom_range`
- `input_atom_range`

By default `project_results_to_ewf` rejects missing block results, failed or
unconverged SIESTA runs, missing DM/HSX/ORB_INDX paths, missing orbital ranges
for core atoms, and matrix headers whose orbital count disagrees with
`ORB_INDX`.  This makes incomplete local calculations fail before they can be
interpreted as global EWF data.

In non-dry-run mode, `ewf_siesta_driver.py` writes both `results.json` and
`ewf_results.json` on rank 0, then writes `global_matrices.json` after
validating core atom ownership and matrix consistency.  Dry-run mode writes
scheduling manifests only.

`schedule.json` is written by rank 0 during setup in both dry-run and real-run
modes.  It records the planned owner rank for every block, including ranks which
have no results yet.  `run_summary.json` is written after result aggregation.
In dry-runs it reports the scheduled block/rank/machine layout; in real runs it
also includes per-block wall time parsed from `siesta.out`, convergence, return
code, core/input sizes, buffer size, and matrix orbital counts.  Together these
manifests are the first weak-scaling comparison surface for different block and
machine counts.

`global_matrices.json` and `validation.json` also include
`density_overlap_trace_total = Tr(D S)` and a per-spin version computed from the
assembled core-owned density and overlap entries.  This is only an electron-count
diagnostic for the current collection surface; it is not a chemical-potential or
electron-number constraint.

After `global_matrices.json` is available, the workflow writes
`electron_constraint.json`.  It estimates the target valence electron count from
`ChemicalSpeciesLabel`, compares it with `density_overlap_trace_total`, and
records the deviation.  The current status is diagnostic only:
`chemical_potential_status=not_applied`.

Multiple run directories can be compared with:

```bash
python3 ewf_weak_scaling_report.py run_a run_b run_c --output weak_scaling_report.json
```

The report uses `max_block_wall_time_seconds` as the per-run wall-time metric and
reports `baseline_time / current_time` as a diagnostic weak-scaling efficiency.
Dry-run entries remain useful for schedule coverage, but their timing and
efficiency fields are `null`.

## Environment Variables

- `EWF_NUM_MACHINES`: simulated machine count, default `1`.
- `EWF_PROCS_PER_MACHINE`: MPI ranks per simulated machine, default `1`.
- `EWF_THREADS_PER_PROC`: BLAS threads per SIESTA process, default `1`.
- `EWF_WORKDIR`: generated run directory, default `runs`.
- `EWF_BLOCK_ATOMS`: core atoms per block. If unset, the adapter chooses one block per simulated worker.
- `EWF_BLOCK_BUFFER_ATOMS`: atom-count buffer on each side, default `0`.
- `EWF_GROUP_SIZE_ATOMS`: optional contiguous atom count per chain/repeat group.
- `EWF_BLOCK_GROUPS`: optional core groups per block. Must be set with `EWF_GROUP_SIZE_ATOMS`.
- `EWF_BLOCK_BUFFER_GROUPS`: optional group-count buffer on each side, default `0`.
- `EWF_TERMINAL_CAP_ATOMS`: optional terminal atoms split between the first and final repeat groups; use `2` for generated finite polyethylene chains.
- `EWF_SIESTA_BIN`: SIESTA executable for non-dry-run execution.
- `EWF_SIESTA_DRY_RUN`: default `true`; set to `false` to execute SIESTA.

Example:

```bash
EWF_NUM_MACHINES=4 \
EWF_PROCS_PER_MACHINE=4 \
EWF_THREADS_PER_PROC=1 \
EWF_BLOCK_ATOMS=1000 \
EWF_BLOCK_BUFFER_ATOMS=100 \
python3 ewf_siesta_driver.py testcases/2306.fdf --pseudo testcases/C.psf --pseudo testcases/H.psf
```

For polyethylene inputs ordered as repeat groups such as `C C H H H H` plus
terminal hydrogens, prefer group-aligned blocks:

```bash
EWF_GROUP_SIZE_ATOMS=6 \
EWF_BLOCK_GROUPS=10 \
EWF_BLOCK_BUFFER_GROUPS=1 \
EWF_TERMINAL_CAP_ATOMS=2 \
python3 ewf_siesta_driver.py testcases/0386.fdf --pseudo testcases/C.psf --pseudo testcases/H.psf
```

For generated finite polyethylene chains, `EWF_TERMINAL_CAP_ATOMS=2` keeps one
terminal atom with the first repeat group and one with the final repeat group,
so terminal hydrogens do not become uncovered nonlocal boundary bonds.

To run SIESTA instead of only generating inputs:

```bash
EWF_SIESTA_BIN=/home/xzz2/huawei-siesta/siesta-install/bin/siesta \
EWF_SIESTA_DRY_RUN=false \
python3 ewf_siesta_driver.py testcases/0386.fdf --pseudo testcases/C.psf --pseudo testcases/H.psf
```

## Local Build Used During Development

The SIESTA/HONPAS source archive was configured with MPI, ScaLAPACK, and ELSI
enabled, using OpenBLAS and the bundled source dependencies:

```bash
cmake -S siesta-honpas-20250306-9346e7056 -B build/siesta-mpi \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/home/xzz2/huawei-siesta/siesta-install \
  -DSIESTA_TESTS=OFF \
  -DSIESTA_WITH_MPI=ON \
  -DSIESTA_WITH_ELSI=ON \
  -DSIESTA_WITH_ELSI_PEXSI=OFF \
  -DSIESTA_WITH_NETCDF=OFF \
  -DSIESTA_WITH_DFTD3=OFF \
  -DSIESTA_WITH_WANNIER90=OFF \
  -DSIESTA_WITH_CHESS=OFF \
  -DSIESTA_WITH_PEXSI=OFF \
  -DSIESTA_WITH_OPENMP=OFF \
  -DBLAS_LIBRARY=openblas \
  -DLAPACK_LIBRARY=openblas \
  -DSCALAPACK_LIBRARY=scalapack-openmpi
cmake --build build/siesta-mpi -j 4
cmake --install build/siesta-mpi
```

The installed executable is:

```text
/home/xzz2/huawei-siesta/siesta-install/bin/siesta
```

## Verified Smoke Test

A complete 8-atom `C2H6` polyethylene fragment generated from `testcases/gen.py`
was run through the adapter and SIESTA.  The calculation converged and produced
the minimum result artifacts needed by the current adapter:

```text
converged=True
energy_ev=-404.898
density_matrix_path=/tmp/ewf-siesta-pe2/block_0000/block_0000.DM
hamiltonian_matrix_path=/tmp/ewf-siesta-pe2/block_0000/block_0000.HSX
overlap_matrix_path=/tmp/ewf-siesta-pe2/block_0000/block_0000.HSX
orbital_index_path=/tmp/ewf-siesta-pe2/block_0000/block_0000.ORB_INDX
atom_orbital_ranges=8 atoms
```

A 14-atom `C4H10` input generated from `testcases/gen.py` was also run as two
group-aligned SIESTA blocks with one group of buffer:

```text
atoms=14
blocks=2
block 0: returncode=0, converged=True, energy_ev=-778.621
block 1: returncode=0, converged=True, energy_ev=-778.621
validation.ok=True
validation.ncore_atoms=14
global_matrices.norbitals=102
global_matrices.nnz=5192
global_matrices.density_overlap_trace_total=24.26909771038243
electron_constraint.target_valence_electrons=26.0
electron_constraint.electron_count_deviation=-1.7309022896175712
```

The same two-block case was run through the outer MPI driver with two ranks:

```text
mpirun -np 2 ...
rank 0 -> block 0 -> converged=True
rank 1 -> block 1 -> converged=True
result_rank_0000.json and result_rank_0001.json written
schedule.block_owner_rank={"0": 0, "1": 1}
boundary.num_boundary_bonds=4
boundary.num_uncovered_boundary_bonds=0
embedding_contract.num_pending_embedding_terms=4
run_summary.ranks_with_results=[0, 1]
run_summary.max_block_wall_time_seconds=2.279
validation.ok=True
global_matrices.norbitals=102
global_matrices.nnz=5192
global_matrices.density_overlap_trace_total=24.26909771038243
electron_constraint.target_valence_electrons=26.0
electron_constraint.electron_count_deviation=-1.7309022896175712
```

This proves the current minimum SIESTA backend path:

```text
global/local FDF -> block input directories -> SIESTA execution -> output reader -> results.json -> ewf_results.json -> global_matrices.json -> validation.json
```

The validation manifest is deliberately conservative.  It checks block result
coverage, convergence, required matrix artifacts, core atom ownership, and sparse
matrix assembly consistency.  Its `total_block_energy_ev` field is only a
diagnostic sum of independent block energies.  Embedded observables and
`predictive_ewf_closure.json` record the current boundary, electron-count, bath,
and mean-field double-counting policies explicitly; production correlated
fragment energy reconstruction is still outside this adapter layer.

If any block SIESTA run fails, the driver now writes `validation.json` with
`ok=false`, keeps the raw rank/result manifests for debugging, skips
`ewf_results.json` and `global_matrices.json`, and exits non-zero without a
Python traceback.
