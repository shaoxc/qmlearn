import os
import numpy as np
from scipy.linalg import eig
from scipy.spatial.transform import Rotation
from ase import Atoms, io
from pyscf.pbc.tools.pyscf_ase import atoms_from_ase
from pyscf import gto, ao2mo
from pyscf import dft, scf, mp, fci, ci, cc, mcscf
from pyscf.symm import Dmatrix
from pyscf.scf import addons
from functools import reduce

from qmlearn.drivers.core import Engine

methods_pyscf = {
        'dft' : dft.RKS,
        'hf' : scf.RHF,
        'rks' : dft.RKS,
        'rhf' : scf.RHF,
        'mp2' : mp.MP2,
        'cisd': ci.CISD,
        'fci' : fci.FCI,
        'ccsd' : cc.CCSD,
        'ccsd(t)' : cc.CCSD,
        'casci' : mcscf.CASCI,
        'casscf' : mcscf.CASSCF,
        }

class EnginePyscf(Engine):
    r""" PySCF calculator

    Attributes
    ----------

    vext : ndarray
        External Potential.
    gamma : ndarray
        1-body reduced density matrix (1-RDM).
    gammat : ndarray
        2-body reduced density matrix (2-RDM).
    etotal : float
        Total electronic energy (Atomic Units a.u.)
    forces : ndarray
        Atomic forces (Atomic Units a.u.)
    ovlp : ndarray
        Overlap Matrix.
    kop : ndarray
        Kinetic Energy Operator.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init()

    def init(self, **kwargs):
        r""" Function to initialize mf PySCF object

        Parameters
        ----------
        mol : :obj: PySCF or ASE atom object
            Molecular geometry. Coordinates in Angstroms.
        basis : dict or str
            To define basis set.
        xc : dict or str
            To define xchange-correlation functional
        verbose :int
            Printing level
        method : str

            | DFT : 'dft'
            | HF : 'hf'
            | RKS : 'rks'
            | RHF : 'rhf
            | MP2 : 'mp2'
            | CISD : 'cisd'
            | FCI : 'fci'

        charge : int
            Total charge of the molecule
        """
        mol = self.options.get('mol', None)
        mf = self.options.get('mf', None)
        basis = self.options.get('basis', '6-31g')
        xc = self.options.get('xc', None) or ''
        verbose = self.options.get('verbose', 0)
        self.method = self.options.get('method', 'rks').lower()
        charge = self.options.get('charge', None)
        #
        if isinstance(self.method, str):
            if self.method in ['mp2', 'cisd', 'fci', 'ccsd', 'ccsd(t)', 'casci', 'casscf'] :
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
        r""" Function to create PySCF atom object

        Parameters
        ----------
        mol : list or str (From PySCF) or ASE atom object
            To define molecluar structure.  The internal format is

            | atom = [[atom1, (x, y, z)],
            |         [atom2, (x, y, z)],
            |         ...
            |         [atomN, (x, y, z)]]

        basis:  dict or str
            To define basis set.

        Returns
        -------
        atoms : :obj: PySCF atom object
           Molecular Structure, basis, and charge definition into gto PySCF atom object
        """
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
            mol = gto.M(atom = atom, basis=basis, charge = charge, parse_arg = False)
        return mol

    def run(self, properties = ('energy', 'forces'), ao_repr = True, **kwargs):
        r""" Caculate electronic properties using PySCF.

        Parameters
        ----------
        properties : str

            | Total electronic energy : 'energy'
            | Total atomic forces : 'forces'

        If 'energy' is choosen the following properties are also calculated:

            | 1-RDM (1-body reduced density matrix) : 'gamma'
            | 2-RDM (2-body reduced density matrix) : 'gammat'
            | Occupation number : 'occs'
            | Molecular orbitals : 'orb'

        """
        if 'energy' in properties or self._gamma is None :
            # (dft.uks.UKS, dft.rks.RKS, scf.uhf.UHF, scf.hf.RHF))
            self.mf.run()
            self.occs = self.mf.get_occ()
            norb = self.occs.shape[0]
            #
            if '+' in self.method :
                method2 = self.method.split('+')[1]
                if method2 == 'fci':
                    mf2 = methods_pyscf[method2](self.mf)
                    mf2.verbose = self.mf.verbose
                    e, ci = mf2.kernel()
                    if 'gammat' in properties :
                        rdm_1, rdm_2 = mf2.make_rdm12(fcivec=ci, norb=norb, nelec=self.mol.nelec)
                        self._gamma, self._gammat = fci.rdm.reorder_rdm(rdm1=rdm_1, rdm2=rdm_2) # Call reorder_rdm to transform to the normal rdm2a
                    else :
                        self._gamma = mf2.make_rdm1(fcivec=ci, norb=norb, nelec=self.mol.nelec)
                    self._orb = self.mf.mo_coeff
                    self._etotal = mf2.e_tot
                elif method2 in ['casci', 'casscf']:
                    ncas = self.options.get('ncas', None)
                    nelecas = self.options.get('nelecas', None)
                    if ncas is None : ncas = norb
                    if nelecas is None : nelecas = self.mol.nelec
                    mf2 = methods_pyscf[method2](self.mf, ncas, nelecas)
                    mf2.verbose = self.mf.verbose
                    # mf2.verbose = 9
                    mf2.kernel()
                    self._gamma = mf2.make_rdm1()
                    self._orb = self.mf.mo_coeff
                    self._etotal = mf2.e_tot
                else :
                    mf2 = methods_pyscf[method2](self.mf)
                    mf2.verbose = self.mf.verbose
                    mf2.run()
                    self._orb = self.mf.mo_coeff
                    self._gamma = mf2.make_rdm1(ao_repr = ao_repr, **kwargs)
                    self._etotal = mf2.e_tot
                    if method2 == 'ccsd(t)':
                        ccsd_t = mf2.ccsd_t()
                        # print('ccst(t)', self._etotal, ccsd_t)
                        self._etotal = self._etotal + ccsd_t
                self.mf2 = mf2
            else :
                mf = self.mf
                self._orb = mf.mo_coeff
                self._gamma = mf.make_rdm1(ao_repr = ao_repr, **kwargs)
                self._etotal = mf.e_tot

        if 'forces' in properties :
            self._forces = self.run_forces()

    @property
    def gamma(self):
        if self._gamma is None:
            self.run(properties = ('energy'))
        return self._gamma

    @property
    def gammat(self):
        if self._gammat is None:
            self.run(properties = ('energy'))
        return self._gammat

    @property
    def etotal(self):
        if self._etotal is None:
            self.run(properties = ('energy'))
        return self._etotal

    @property
    def forces(self):
        if self._forces is None:
            self.run(properties = ('forces'))
        return self._forces

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
        r""" Total number of electrons. """
        return self.mol.nelectron

    @property
    def nao(self):
        r""" Natural atomic orbitals. """
        return self.mol.nao

    def calc_gamma(self, orb=None, occs=None):
        if orb is None : orb = self.orb
        if occs is None : orb = self.occs
        nm = min(orb.shape[1], len(occs))
        gamma = self.mf.make_rdm1(orb[:, nm], occs[:nm])
        return gamma

    def calc_etotal(self, gamma, **kwargs):
        r""" Get the total electronic energy based on 1-RDM.

        Parameters
        ----------
        gamma : ndarray
            1-RDM

        Returns
        -------
        etotal : float
            Total electronic energy. """
        etotal = self.mf.energy_tot(gamma, **kwargs)  # nuc + energy_elec(e1+coul)
        return etotal

    def calc_etotal2(self, gammat, gamma1=None, **kwargs):
        r""" Get the total electronic energy based on 2-RDM.

        Parameters
        ----------
        gamma : ndarray
            1-RDM
        gammat : ndarray
            2-RDM

        Returns
        -------
        etotal : float
            Total electronic energy. """

        self.mf.run() # HF to get orb and occ
        orb = self.mf.mo_coeff
        occs = self.mf.get_occ()

        nmo = len(occs)
        h1e = reduce(np.dot,(orb.T, self.mf.get_hcore(), orb))
        h2e = ao2mo.kernel(self.mf._eri, orb)
        h2e = ao2mo.restore(1, h2e, nmo)

        etotal = (np.einsum('ij,ji', h1e, gamma1) + np.einsum('ijkl,ijkl', h2e, gammat) * .5)
        #etotal = (self.mf.energy_elec(gamma1) + np.einsum('ijkl,ijkl', h2e, gammat) * .5)
        #etotal = (np.einsum('ijkl,ijkl', h2e, gammat) * .5)
        etotal += self.mol.energy_nuc()

        return etotal

    def calc_dipole(self, gamma, **kwargs):
        r""" Get the total dipole moment.

        Parameters
        ----------
        gamma : ndarray
            1-RDM

        Returns
        -------
        dip : list
            The dipole moment on x, y and z component."""
        dip = self.mf.dip_moment(self.mol, gamma, unit = 'au', verbose=self.mf.verbose)
        return dip

    def run_forces(self, **kwargs):
        r""" Function to calculate Forces with calculated 1-RDM

        Returns
        -------
        forces : ndarray
           Total atomic forces. """
        if '+' in self.method :
            if self.method.split('+')[1] in ['fci']:  # With FCI method Forces can't be obtained from FCI object. I approximated using RHF object.
                mf = self.mf
            else:
                mf = self.mf2
        else :
            mf = self.mf

        gf = mf.nuc_grad_method()
        gf.verbose = mf.verbose
        if hasattr(gf, 'grid_response') :
            gf.grid_response = True
        forces = -1.0 * gf.kernel()
        return forces

    def calc_forces(self, gamma, **kwargs):
        r""" Function to calculate Forces with a given 1-RDM

        Parameters
        ----------
        gamma : ndarray
            1-RDM

        Returns
        -------
        forces : ndarray
           Total atomic forces for a given 1-RDM. """

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
        r""" Function to calculate the total quadruple for XX, XY, XY, YY, YZ, ZZ components.

        Parameters
        ----------
        gamma : ndarray
            1-RDM

        Returns
        -------
        quadrupol : ndarray
           An array containing a quadruple per component."""

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
        r""" Function to calculate the nuclear quadruple for XX, XY, XY, YY, YZ, ZZ components.

        Parameters
        ----------
        gamma : ndarray
            1-RDM

        Returns
        -------
        quadrupol : ndarray
           An array containing the nuclear quadruple per component."""

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

    def rotation2rotmat(self, rotation, mol = None):
        r""" Function to rotate the density matrix.

        Parameters
        ----------
        rotation : ndarray
            Rotation Matrix
        mol : :obj: PySCF mol object
            Molecluar structure

        Returns
        -------
        rotmat : ndarray
        Rotated density matrix """

        mol = mol or self.mol
        return rotation2rotmat(rotation, mol)

    def get_atom_naos(self, mol = None):
        mol = mol or self.mol
        return get_atom_naos(mol)

    def project_dm_nr2nr(self, gamma, mol2, mol = None):
        mol = mol or self.mol
        if hasattr(mol, 'mol'): mol = mol.mol
        if hasattr(mol2, 'mol'): mol2 = mol2.mol
        dm = addons.project_dm_nr2nr(mol, gamma, mol2)
        return dm

def gamma2gamma(*args, **kwargs):
    r""" Function two assure 1-RDM to be the predicted one.

    Returns
    -------
    gamma : ndarray
        1-RDM """
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
    r""" Function to calculate the energy density matrix (1-RDMe).

    .. math::
        \hat{D} = \sum_{i=1}^{occ} \epsilon_{i} \left|i\right> \left< i \right| \\
        \left< \mu |\hat{D}| \nu \right> = \sum_{\sigma \tau} F_{\mu \sigma} S_{\sigma \tau}^{-1} P_{\tau \mu}

    Parameters
    ----------
    mf : :obj: PySCF object
        SCF class of PySCF

    Returns
    -------
    dm1e : ndarray
        Energy density matrix """
    gamma = gamma2gamma(*args, **kwargs)
    s = mf.get_ovlp()
    sinv = np.linalg.inv(s)
    f = mf.get_fock(dm = gamma)
    dm1e = sinv@f@gamma
    return dm1e

def rotation2rotmat(rotation, mol):
    r""" Function to rotate the density matrix.

    Parameters
    ----------
    rotation : ndarray
        Rotation Matrix

    Returns
    -------
    rotmat : ndarray
        Rotated density matrix """
    angl = []
    for ib in range(mol.nbas):
        l = mol.bas_angular(ib)
        nc = mol.bas_nctr(ib)
        for n in range(nc): angl.append(l)
    angl=np.asarray(angl)
    dims = angl*2+1
    aol = np.empty(len(dims)+1, dtype=np.int32)
    aol[0] = 0
    dims.cumsum(dtype=np.int32, out=aol[1:])
    rotmat = np.eye(mol.nao)
    if np.allclose(rotation, np.eye(len(rotation))) :
        pass
    else :
        alpha, beta, gamma = Rotation.from_matrix(rotation).as_euler('zyz')*-1.0
        for i in range(len(angl)):
            r = Dmatrix.Dmatrix(angl[i], alpha, beta, gamma, reorder_p=True)
            rotmat[aol[i]:aol[i+1],aol[i]:aol[i+1]]=r
    return rotmat

def get_atom_naos(mol):
    r""" Function to get the number of atomic orbitals for a given angular momentum.

    Parameters
    ----------
    mol : :obj: PySCF mol object
        Molecluar structure.

    Returns
    -------
    naos : ndarray
        Number of atomic orbitals. """
    angl = []
    atomids = []
    for ib in range(mol.nbas):
        l = mol.bas_angular(ib)
        nc = mol.bas_nctr(ib)
        ia = mol.bas_atom(ib)
        for n in range(nc):
            angl.append(l)
            atomids.append(ia)
    atomids = np.asarray(atomids)
    angl = np.asarray(angl)
    dims = angl*2+1
    naos = []
    for ia in range(mol.natm) :
        n = np.sum(dims[atomids == ia])
        naos.append(n)
    naos = np.asarray(naos)
    return naos
