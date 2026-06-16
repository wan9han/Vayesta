"""energy_first — energy-reproducing fragmentation prototype (Stage 0).

Goal reference (energy_first_plan.md, §0):
    Use an energy-reproducible fragmentation method to cut a polyethylene
    system into large contiguous blocks, run each block in SIESTA, and
    combine the block energies so the result matches the full-system
    SIESTA energy, while keeping each local block size-bounded for weak
    scaling to ~10k machines.

Stage 0 scope: prove the *combination formulas* are mathematically correct
with a mock backend (no SIESTA, no physics). A mock backend cannot be
"cheated": given defined subsystem energies, the formulas must reproduce
the defined totals to machine precision.

Two combination primitives are provided:

* :func:`~energy_first.mbe.many_body_expansion` — many-body expansion
  (inclusion–exclusion). Exact: at ``max_order == num_fragments`` the
  reconstructed total equals the full-system energy to ~1e-12. This is the
  non-covalent / interaction recombination and the foundation of every
  fragment method.

* :func:`~energy_first.mfcc.mfcc_total` — MFCC-style cap subtraction
  ``E = Σ capped_fragment − Σ conjugate_cap``. Linear. This is how a
  *covalent* cut (e.g. a polyethylene C–C bond) is repaired: each fragment
  is capped into a closed-shell molecule and the cap energies are removed.

Polyethylene is covalent, so Stage 1 will use MFCC caps; MBE handles the
residual many-body interaction between fragments. Both must be correct
before any SIESTA number is trusted.
"""

from .mbe import (
    compute_increments,
    many_body_expansion,
    total_energy,
)
from .mfcc import mfcc_total
from .mock import MockBackend
from .subsystems import iter_subsystems

__all__ = [
    "compute_increments",
    "many_body_expansion",
    "total_energy",
    "mfcc_total",
    "MockBackend",
    "iter_subsystems",
]
