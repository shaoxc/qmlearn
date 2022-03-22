import os
import numpy as np
from scipy.linalg import eig
from ase import Atoms, io
from pyscf.pbc.tools.pyscf_ase import atoms_from_ase
from pyscf import gto, dft, scf, mp, fci, ci

from qmlearn.drivers.core import Engine

methods_pyscf = {
        'dft' : dft.RKS,
        'hf' : scf.RHF,
        'rks' : dft.RKS,
        'rhf' : scf.RHF,
        'mp2' : mp.MP2,
        'cisd': ci.CISD,
        'fci' : fci.FCI,
        }

class EnginePyscf(Engine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init(self, **kwargs):
        mol = self.options.get('mol', None)
        mf = self.options.get('mf', None)
        basis = self.options.get('basis', '6-31g')
        xc = self.options.get('xc', None) or ''
        verbose = self.options.get('verbose', 0)
        self.method = self.options.get('method', 'rks').lower()
        charge = self.options.get('charge', None)
        #
        if isinstance(self.method, str):
            if self.method in ['mp2', 'cisd', 'fci'] :
                self.method = 'rhf+' + self.method
        if self.method.count('+') > 1 :
            raise AttributeError("Sorry, only support two methods at the same time.")
        for fm in self.method.split('+') :
            if fm not in methods_pyscf :
                raise AttributeError(f"Sorry, not support the {fm} method now")
        if mf is None :
            mol = self.init_mol(mol, basis=basis, charge = charge)
            mf = methods_pyscf[self.method.split('+')[0]](mol)
            mf.xc = xc
            mf.verbose = verbose
        self.mf = mf
        self.mol = self.mf.mol

    def init_mol(self, mol, basis, charge = 0):
        ase_atoms = None
        if isinstance(mol, Atoms):
            ase_atoms = mol
        if isinstance(mol, (str, os.PathLike)):
            ase_atoms = io.read(mol)
        else:
            atom = mol
        if ase_atoms is not None :
            atom = atoms_from_ase(ase_atoms)

        if isinstance(atom, gto.Mole):
            mol = atom
        else :
            mol = gto.M(atom = atom, basis=basis, charge = charge)
        return mol

    def run(self, ao_repr = True, **kwargs):
        # (dft.uks.UKS, dft.rks.RKS, scf.uhf.UHF, scf.hf.RHF))
        self.mf.run()
        self.occs = self.mf.get_occ()
        #
        if '+' in self.method :
            mf2 = methods_pyscf[self.method.split('+')[1]](self.mf)
            mf2.verbose = self.mf.verbose
            self.mf2 = mf2.run()
            mf = self.mf2
        else :
            mf = self.mf

        self.orb = mf.mo_coeff
        self.gamma = mf.make_rdm1(ao_repr = ao_repr, **kwargs)
        self.etotal = mf.e_tot
        self.forces = self.run_forces()

    @property
    def vext(self):
        if self._vext is None:
            self._vext = self.mol.intor_symmetric('int1e_nuc')
        return self._vext

    @property
    def kop(self):
        if self._kop is None:
            self._kop = self.mol.intor_symmetric('int1e_kin')
        return self._kop

    @property
    def ovlp(self):
        if self._ovlp is None:
            self._ovlp = self.mol.intor_symmetric('int1e_ovlp')
        return self._ovlp

    @property
    def nelectron(self):
        return self.mol.nelectron

    def calc_gamma(self, orb=None, occs=None):
        if orb is None : orb = self.orb
        if occs is None : orb = self.occs
        nm = min(orb.shape[1], len(occs))
        gamma = self.mf.make_rdm1(orb[:, nm], occs[:nm])
        return gamma

    def calc_etotal(self, gamma, **kwargs):
        etotal = self.mf.energy_tot(gamma, **kwargs)
        return etotal

    def calc_dipole(self, gamma, **kwargs):
        dip = self.mf.dip_moment(self.mol, gamma, unit = 'au', verbose=self.mf.verbose)
        return dip

    def run_forces(self, **kwargs):
        if '+' in self.method :
            mf = self.mf2
        else :
            mf = self.mf
        gf = mf.nuc_grad_method()
        gf.verbose = mf.verbose
        gf.grid_response = True
        forces = -1.0 * gf.kernel()
        return forces

    def calc_forces(self, gamma, **kwargs):
        # only for rhf and rks
        if self.method.split('+')[0] not in ['rhf', 'rks'] or '+' in self.method :
            raise AttributeError(f"Sorry the calc_forces not support '{self.method}'")
        gf = self.mf.nuc_grad_method()
        gf.verbose = self.mf.verbose
        gf.grid_response = True
        # Just a trick to skip the make_rdm1 and make_rdm1e without change the kernel
        save_make_rdm1, self.mf.make_rdm1 = self.mf.make_rdm1, gamma2gamma
        save_make_rdm1e, gf.make_rdm1e = gf.make_rdm1e, gamma2rdm1e
        forces = -1.0 * gf.kernel(mo_energy=self.mf, mo_coeff=gamma, mo_occ=gamma)
        self.mf.make_rdm1 = save_make_rdm1
        gf.make_rdm1e = save_make_rdm1e
        return forces

    def _calc_quadrupole_r(self, gamma, component = 'zz'):
        if component != 'zz' :
            raise AttributeError("Sorry, only support 'zz' component now")
        r2 = self.mol.intor_symmetric('int1e_r2')
        z2 = self.mol.intor_symmetric('int1e_zz')
        q = 1/2*(3*z2 - r2)
        quad = np.einsum('mn, mn->', gamma, q)
        return quad

    def calc_quadrupole(self, gamma, traceless = True):
        # XX, XY, XZ, YY, YZ, ZZ
        with self.mol.with_common_orig((0,0,0)):
            q = self.mol.intor('int1e_rr', comp=9).reshape((3,3,*gamma.shape))
        quadp = np.zeros(6)
        quadp[0] = np.einsum('ij,ij->', q[0, 0], gamma)
        quadp[1] = np.einsum('ij,ij->', q[0, 1], gamma)
        quadp[2] = np.einsum('ij,ij->', q[0, 2], gamma)
        quadp[3] = np.einsum('ij,ij->', q[1, 1], gamma)
        quadp[4] = np.einsum('ij,ij->', q[1, 2], gamma)
        quadp[5] = np.einsum('ij,ij->', q[2, 2], gamma)
        quadp *= -1.0
        quadp += self.calc_quadrupole_nuclear()
        if traceless :
            trace = (quadp[0] + quadp[3] + quadp[5])/3.0
            quadp[0] -= trace
            quadp[3] -= trace
            quadp[5] -= trace
        return quadp

    def calc_quadrupole_nuclear(self, mol = None):
        mol = mol or self.mol
        charges = mol.atom_charges()
        pos = mol.atom_coords()
        nucquad = np.zeros(6)
        nucquad[0] = np.sum(charges * pos[:, 0] * pos[:, 0])
        nucquad[1] = np.sum(charges * pos[:, 0] * pos[:, 1])
        nucquad[2] = np.sum(charges * pos[:, 0] * pos[:, 2])
        nucquad[3] = np.sum(charges * pos[:, 1] * pos[:, 1])
        nucquad[4] = np.sum(charges * pos[:, 1] * pos[:, 2])
        nucquad[5] = np.sum(charges * pos[:, 2] * pos[:, 2])
        return nucquad

    def _get_loc_orb(self, nocc=-1):
        r2 = self.mol.intor_symmetric('int1e_r2')
        r = self.mol.intor_symmetric('int1e_r', comp=3)
        M = np.einsum('mi,nj,mn->ij', self.mf.mo_coeff[:,:nocc], self.mf.mo_coeff[:,:nocc], r2)
        J = np.einsum('mi,nj,tmn->tij', self.mf.mo_coeff[:,:nocc], self.mf.mo_coeff[:,:nocc], r)**2
        M -= np.einsum('tij-> ij', J)
        w,vr = eig(M)
        mo = self.mf.mo_coeff[:,:nocc] @ vr.T
        return mo

def gamma2gamma(*args, **kwargs):
    gamma = None
    for v in args :
        if isinstance(v, np.ndarray):
            gamma = v
            break
    if gamma is None :
        for k, v in kwargs :
            if isinstance(v, np.ndarray):
                gamma = v
                break
    if gamma is None :
        raise AttributeError("Please give one numpy.ndarray for gamma.")
    return gamma

def gamma2rdm1e(mf, *args, **kwargs):
    gamma = gamma2gamma(*args, **kwargs)
    s = mf.get_ovlp()
    sinv = np.linalg.inv(s)
    f = mf.get_fock(dm = gamma)
    dm1e = sinv@f@gamma
    return dm1e
