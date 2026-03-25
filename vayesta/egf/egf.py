# --- Standard
import dataclasses
from multiprocessing import Value

# --- External
import numpy as np
from pyscf.data.nist import HARTREE2EV
from vayesta.core.util import (
    NotCalculatedError,
    break_into_lines,
    cache,
    deprecated,
    dot,
    einsum,
    energy_string,
    log_method,
    log_time,
    time_string,
    timer,
)
from vayesta.core.qemb import Embedding
from vayesta.core.fragmentation import SAO_Fragmentation
from vayesta.core.fragmentation import IAOPAO_Fragmentation
from vayesta.core.qemb.static_observable import make_global_one_body, make_local_one_body, symmetrize_observable
from vayesta.core.types.dynamical import SE_LehmannRep
from vayesta.mpi import mpi
from vayesta.ewf.ewf import REWF
from vayesta.egf.fragment import Fragment
from vayesta.egf.self_energy import *
from vayesta.egf.qsegf import QSEGF_RHF

from dyson import MBLGF, MBLSE, FCI, CCSD, AufbauPrinciple, AuxiliaryShift, Lehmann
from dyson.solvers.static.chempot import search_aufbau_global as find_chempot
from dyson.util.moments import se_moments_to_gf_moments, gf_moments_to_se_moments


@dataclasses.dataclass
class Options(REWF.Options):
    """Options for EGF calculations."""
    proj: int = 1     # Number of projectors used on self-energy (1, 2)
    proj_static_se: int = 1 # Number of projectors used on static self-energy (same as proj if None)
    use_sym: bool = True # Use symmetry for self-energy reconstruction

    
    chempot_global: str = None # Use auxiliary shift to ensure correct electron number in the physical space (None, 'auf', 'aux')
    chempot_clus: str = 'auto' # Use auxiliary shift to ensure correct electron number in the fragment space (None, 'auf', 'aux')
    se_mode: str = 'moments_mblgf' # Mode for self-energy reconstruction (moments, lehmann)
    static_se_mode: str = 'cluster_moments_corr' # Method for static self-energy (cluster_moments, fock, fock_corr, cluster_fock_corr)
    nmom_se: int = None      # Number of conserved moments for self-energy
    global_1dm: bool = False # Use global 1DM to normalise SE moments
    sym_moms: bool = True # Use symmetrized moments
    hermitian_lanczos: bool = False # Use hermitian Lanczos algorithm 
    img_space : bool = True    # Use image space for self-energy reconstruction
    drop_non_causal: bool = False # Drop non-causal poles
    se_degen_tol: float = 1e-6 # Tolerance for degeneracy of self-energy poles
    se_eval_tol: float = 1e-6  # Tolerance for self-energy eignvalues

    non_local_se: str = None # Non-local self-energy (GW, CCSD, FCI)
    se_dc_mode: str = 'global' # Mode for double counting correction of non-local self-energy (local, global)


    solver_options: dict = Embedding.Options.change_dict_defaults("solver_options", n_moments=(6,6), conv_tol=1e-15, conv_tol_normt=1e-15)
    bath_options: dict = Embedding.Options.change_dict_defaults("bath_options", bathtype='ewdmet', order=1, max_order=1, dmet_threshold=1e-12)

class REGF(REWF):
    Options = Options
    Fragment = Fragment

    def __init__(self, mf, solver="CCSD", log=None, **kwargs):
        super().__init__(mf, solver=solver, log=log, **kwargs)

        if self.opts.proj_static_se is None:
            self.opts.proj_static_se = self.opts.proj

        if self.opts.chempot_clus == 'auto':
            if self.opts.se_mode == 'moments' or self.opts.se_mode == 'moments_mblgf':
                self.opts.chempot_clus = 'aux' if self.opts.chempot_global == 'aux' else 'auf'
            elif self.opts.se_mode == 'lehmann':
                self.opts.chempot_clus = None
            elif self.opts.se_mode == 'spectral':
                self.opts.chempot_clus = 'aux' if self.opts.chempot_global == 'aux' else None
            else:
                raise ValueError("Invalid self-energy mode")
            
        # Logging
        with self.log.indent():
            # Options
            self.log.info("Parameters of %s:", self.__class__.__name__)
            self.log.info(break_into_lines(str(self.opts), newline="\n    "))
            #self.log.info("Time for %s setup: %s", self.__class__.__name__, time_string(timer() - t0))
    
    def kernel(self, run_ewf=True):
        """Run the EGF calculation"""

        if run_ewf:
            super().kernel()
       
        with log_time(self.log.info, "Time for self-energy: %s"):   
            self.se_rep = self.make_self_energy(se_mode=self.opts.se_mode, hermitian_lanczos=self.opts.hermitian_lanczos, proj=self.opts.proj)

        with log_time(self.log.info, "Time for Green's function: %s"):
            self.gf, self.se = self.make_greens_function(self.se_rep, chempot_global=self.opts.chempot_global)
        
        #gm_energy = self.galitskii_migdal(self.gf, self.se)
        #self.log.info("Galitskii-Migdal energy: %s", energy_string(gm_energy))

        ea = self.gf.physical().virtual().energies[0]
        ip = self.gf.physical().occupied().energies[-1]
        self.log.info("Quasiparticle energies from Green's function poles:")
        if ip.imag > 1e-6 or ea.imag > 1e-6:
            self.log.warning("Warning: Significant imaginary part in GF energies: IP = %f + %fj, EA = %f + %fj"%(ip.real, ip.imag, ea.real, ea.imag))
        ip, ea = ip.real, ea.real
        self.log.info("IP: %8f   EA: %8f   Gap: %8f Ha"%(ip.real, ea.real, (ea-ip).real))
        ip, ea = ip * HARTREE2EV, ea * HARTREE2EV
        self.log.info("IP: %8f   EA: %8f   Gap: %8f eV"%(ip, ea, (ea-ip)))

        ip = self.gf.as_perturbed_mo_energy()[self.mf.mol.nelectron//2-1] 
        ea = self.gf.as_perturbed_mo_energy()[self.mf.mol.nelectron//2] 
        self.log.info("Quasiparticle energies from MO overlap:")
        if ip.imag > 1e-6 or ea.imag > 1e-6:
            self.log.warning("Warning: Significant imaginary part in MO overlap energies: IP = %f + %fj, EA = %f + %fj"%(ip.real, ip.imag, ea.real, ea.imag))
        ip, ea = ip.real, ea.real
        self.log.info("IP: %8f   EA: %8f   Gap: %8f Ha"%(ip.real, ea.real, (ea-ip).real))
        ip, ea = ip * HARTREE2EV, ea * HARTREE2EV
        self.log.info("IP: %8f   EA: %8f   Gap: %8f eV"%(ip, ea, (ea-ip)))

    def make_greens_function(self, se, chempot_global=None):
        """
        Calculate Green's function from self-energy using Dyson equation.

        Parameters
        ----------
        se : SE_Lehmann
            Self-energy in Lehmann representation
        chempot_global : (None, 'auf', 'aux')
            Type of chemical potential optimisation for full system Green's function. AufbauPrinciple or AuxiliaryShift.
        
        Returns
        -------
        gf : Lehmann
            Green's function
        """

        if chempot_global is None:
           chempot_global = self.opts.chempot_global

        assert se.statics.ndim == 2
        assert se.overlaps is None
        static = se.statics

        if isinstance(se, SE_LehmannRep):
            if se.nsectors > 1:
                se = se.combine_sectors()
            self_energy = se.lehmanns[0]
            self.log.info("Diagonalsing Lehmann SE with nphys = %d, naux = %d"%(self_energy.couplings.shape[-2:]))
            gf = Lehmann(*self_energy.diagonalise_matrix_with_projection(static) )

        elif isinstance(se, SE_MomentRep):
            assert se.nsectors == 2
            self.log.info("Running MBLSE with %d (hole/particle) self-energy moments"%(se.moments.shape[1]))
            res = []
            for i, s in enumerate(se.moments):
                moms = se.moments[i]
                solver = MBLSE(static, moms, hermitian=self.opts.hermitian_lanczos)
                solver.kernel()
                res.append(solver.result)
            res = dyson.Spectral.combine_for_self_energy(*res)
            self_energy = res.get_self_energy()
            self.log.info("Diagonalsing Lehmann SE with nphys = %d, naux = %d"%(self_energy.couplings.shape))
            spec = dyson.Spectral.from_self_energy(static, self_energy)
            gf = spec.get_greens_function()
            self_energy = spec.get_self_energy()

        else:
            raise NotImplementedError("Green's function construction not implemented for self-energy type %s"%type(se))
    
        # Add fock self-consistency here?

        if chempot_global == 'auf':
            cpt, err = find_chempot(gf, self.mf.mol.nelectron, occupancy=2)
            gf = gf.copy(chempot=cpt)
            self.log.info("Aufbau chemical potential shift: %f (error in N_elec: %e)"%(cpt, err))

        elif chempot_global == 'aux':
            mu_solver = dyson.solvers.static.chempot.AuxiliaryShift(static, self_energy, self.mf.mol.nelectron)
            mu_solver.kernel()
            result = mu_solver.result
            gf = result.get_greens_function()
            self_energy = result.get_self_energy()
            self.log.info("Auxiliary chemical potential shift: %f "%(result.chempot))

        dm = gf.occupied().moment(0) 
        nelec_gf = np.trace(dm) * 2.0
        self.log.info('Number of electrons in GF: %f'%nelec_gf)
        return gf, self_energy


    def make_self_energy(self, se_mode=None, static_se_mode=None, hermitian_lanczos=None, proj=None, proj_static_se=None):

        se_mode = self.opts.se_mode if se_mode is None else se_mode
        static_se_mode = self.opts.static_se_mode if static_se_mode is None else static_se_mode
        hermitian_lanczos = self.opts.hermitian_lanczos if hermitian_lanczos is None else hermitian_lanczos
        proj = self.opts.proj if proj is None else proj
        proj_static_se = self.opts.proj_static_se if proj_static_se is None else proj_static_se

        # Build static self-energy
        dm1 = None
        self.log.info("Calculating static self-energy with %s method."%static_se_mode)
        if static_se_mode in ['cluster_moments', 'cluster_moments_corr', 'cluster_fock_corr', 'global_fock_corr']:
            se_static = make_static_self_energy(self, 
                                                proj=proj_static_se, 
                                                sym_moms=self.opts.sym_moms, 
                                                static_se_mode=static_se_mode,
                                                use_sym=self.opts.use_sym)
            
        elif static_se_mode == 'fock':
            se_static = np.diag(self.mf.mo_energy)
        elif static_se_mode == 'fock_corr':
            dm1 = self._make_rdm1_ccsd_global_wf(self, slow=True, ao_basis=True) if dm1 is not None else dm1
            fock_corr_ao = self.mf.get_fock(dm=dm1) 
            se_static = self.mo_coeff.T @ fock_corr_ao @ self.mo_coeff
        else:
            raise ValueError("Invalid static self-energy mode: %s"%static_se_mode)
        
        assert se_static.ndim == 2, "Static self-energy should be a 2D array"


        if self.opts.non_local_se is not None:
            s = "GW(TDA)" if self.opts.non_local_se == 'gw_tda' else "GW(RPA)"
            m = "cluster" if self.opts.se_dc_mode == 'cluster' else 'global'
            nl = " with non-local %s self-energy using %s DC correction"%(s, m)
        else:
            nl = '.'
        self.log.info("Calculating dynamic self-energy with %s projectors, using %s method%s", self.opts.proj, self.opts.se_mode, nl)  

        se = make_self_energy(self, 
                              se_mode=self.opts.se_mode, 
                              chempot_clus=self.opts.chempot_clus,
                              proj=self.opts.proj,
                              non_local_se=self.opts.non_local_se, 
                              se_dc_mode=self.opts.se_dc_mode, 
                              hermitian=self.opts.hermitian_lanczos)
        
        # Overwrite static, with static from selected method
        se._statics = se_static
        # Overlap should always be None
        se._overlaps = None
        
        #assert se.hermitian == hermitian_lanczos, "Hermiticity of self-energy does not match specified value"
    
        return se
    

    def make_static_self_energy(self, proj, sym_moms=False, with_mf=False, use_sym=True):
        return make_static_self_energy(self, proj=proj, sym_moms=sym_moms, with_mf=with_mf, use_sym=use_sym)

    def make_gf_moments(self, nmom=2):

        fragments = self.get_fragments(sym_parent=None) if self.opts.use_sym else self.get_fragments()
        # overlap and static self-energy are first two GF moments
        gf_moms_clusters = [f.results.gf_moments[:,:nmom,:,:] for f in fragments]
        gf_moms = make_global_one_body(self, gf_moms_clusters, symmetrize=sym_moms, use_sym=self.opts.use_sym, proj=1, fragments=fragments)
        return gf_moms
    

    def qsEGF(self, *args, **kwargs):
        """Convert EGF to qsEGF"""
        self.with_scmf = QSEGF_RHF(self, *args, **kwargs)
        self.kernel = self.with_scmf.kernel



