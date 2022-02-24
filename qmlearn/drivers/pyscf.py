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
        xc = self.options.get('xc', None)
        verbose = self.options.get('verbose', 0)
        self.method = self.options.get('method', 'rks').lower()
        #
        if isinstance(self.method, str):
            if self.method in ['mp2', 'cisd', 'fci'] :
                self.method = 'rhf+' + self.method
        if mf is None :
            mol = self.init_mol(mol, basis=basis)
            if self.method.count('+') > 1 :
                raise AttributeError("Sorry, only support two methods at the same time.")
            for fm in self.method.split('+') :
                if fm not in methods_pyscf :
                    raise AttributeError(f"Sorry, not support the {fm} method now")
            mf = methods_pyscf[self.method.split('+')[0]](mol)
            mf.xc = xc
            mf.verbose = verbose
        self.mf = mf
        self.mol = self.mf.mol

    def init_mol(self, mol, basis):
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
            mol = gto.M(atom = atom, basis=basis)
        return mol

    def run(self, ao_repr = True, **kwargs):
        mf0 = getattr(self.mf, '_scf', None)
        if isinstance(mf0, (dft.uks.UKS, dft.rks.RKS, scf.uhf.UHF, scf.hf.RHF)):
            self.mf._scf.run()
        #
        self.mf.run()
        #
        if '+' in self.method and mf0 is None :
            mf2 = methods_pyscf[self.method.split('+')[1]](self.mf)
            self.mf = mf2.run()
        try:
            self.occs = self.mf.get_occ()
        except Exception :
            self.occs = self.mf._scf.get_occ()
        self.orb = self.mf.mo_coeff
        self.gamma = self.mf.make_rdm1(ao_repr = ao_repr, **kwargs)
        self.etotal = self.mf.e_tot

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
        try:
            etotal = self.mf.energy_tot(gamma, **kwargs)
        except Exception :
            etotal = self.mf._scf.energy_tot(gamma, **kwargs)
        return etotal

    def calc_dipole(self, gamma, **kwargs):
        try:
            dip = self.mf.dip_moment(self.mol, gamma)
        except Exception :
            dip = self.mf._scf.dip_moment(self.mol, gamma)
        return dip

    def calc_quadrupole(self, gamma, component = 'zz'):
        if component != 'zz' :
            raise AttributeError("Sorry, only support 'zz' component now")
        r2 = self.mol.intor_symmetric('int1e_r2')
        z2 = self.mol.intor_symmetric('int1e_zz')
        q = 1/2*(3*z2 - r2)
        quad = np.einsum('mn, mn->', gamma, q)
        return quad

    def _get_loc_orb(self, nocc=-1):
        r2 = self.mol.intor_symmetric('int1e_r2')
        r = self.mol.intor_symmetric('int1e_r', comp=3)
        M = np.einsum('mi,nj,mn->ij', self.mf.mo_coeff[:,:nocc], self.mf.mo_coeff[:,:nocc], r2)
        J = np.einsum('mi,nj,tmn->tij', self.mf.mo_coeff[:,:nocc], self.mf.mo_coeff[:,:nocc], r)**2
        M -= np.einsum('tij-> ij', J)
        w,vr = eig(M)
        mo = self.mf.mo_coeff[:,:nocc] @ vr.T
        return mo
