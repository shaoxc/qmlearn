import numpy as np
from ase import Atoms, io

from qmlearn.drivers.core import minimize_rmsd_operation
from qmlearn.utils import array2blocks, blocks2array

qm_engines = {
        'pyscf' : None
        }

class QMMol(object):
    r""" Class to create qmlearn mol object.

    Attributes
    ----------
    atoms : :obj: PySCF or ASE atom object
        Molecular geometry
    engine_name : str
        Options:

        | PySCF (Default) : 'pyscf'
        | Psi4 : 'psi4'

    method : str

        | PySCF:
        | DFT : 'dft'
        | HF : 'hf'
        | RKS : 'rks'
        | RHF : 'rhf
        | MP2 : 'mp2'
        | CISD : 'cisd'
        | FCI : 'fci'

        | Psi4:
        | HF('rhf+hf')  : 'hf'
        | RHF('rhf+hf') : 'rhf
        | SCF('rks+scf') : 'scf'
        | RKS('rks+scf') : 'rks'
        | MP2('rhf+mp2') : 'mp2'

    basis : dict or str
        To define basis set.
    xc : dict or str
        To define xchange-correlation functional
    charge : int
        Total electronic charge.
    rotate_method : str

        | None : 'none'
        | Kabsch : 'kabsch'
        | Quaternion : 'quaternion'

    reorder_method : str

        | 'hungarian'
        | 'inertia-hungarian'
        | 'brute'
        | 'distance'

    use_reflection : bool
        If True it applies a reflection on your molecule.

    """

    engine_calcs = [
            'calc_gamma',
            'calc_ncharge',
            'calc_etotal',
            'calc_etotal2',
            'calc_ke',
            'calc_dipole',
            'calc_quadrupole',
            'calc_forces',
            'calc_idempotency',
            'calc_occupations',
            'purify_gamma',
            'rotation2rotmat',
            'get_atom_naos',
            'vext',
            'ovlp',
            'ovlp_x',
            'ovlp_x_inv',
            'nao',
            ]

    def __init__(self, atoms = None, engine_name = 'pyscf', method = 'rks', basis = '6-31g',
            xc = None, occs=None, refatoms = None, engine_options = {}, charge = None, engine = None,
            stereo = True, rotate_method = 'kabsch', reorder_method = 'none', use_reflection = True,
            ignore_hydrogen = False, **kwargs):
        # Save all the kwargs for duplicate
        self.init_kwargs = locals()
        self.init_kwargs.pop('self', None)
        self.init_kwargs.pop('kwargs', None)
        self.init_kwargs.update(kwargs)
        self.init()
        #
        for key in self.engine_calcs :
            setattr(self, key, getattr(self.engine, key))

    def init(self):
        # Draw the arguments
        engine=self.init_kwargs.get('engine', None)
        atoms=self.init_kwargs.get('atoms', None)
        occs=self.init_kwargs.get('occs', None)
        refatoms=self.init_kwargs.get('refatoms', None)
        #
        engine_name=self.init_kwargs.get('engine_name', None)
        engine_options=self.init_kwargs.get('engine_options', {}).copy()
        method=self.init_kwargs.get('method', None)
        xc=self.init_kwargs.get('xc', None)
        basis=self.init_kwargs.get('basis', None)
        charge=self.init_kwargs.get('charge', None)
        #
        stereo=self.init_kwargs.get('stereo', True)
        rotate_method=self.init_kwargs.get('rotate_method', None)
        reorder_method=self.init_kwargs.get('reorder_method', None)
        use_reflection=self.init_kwargs.get('use_reflection', True)
        ignore_hydrogen=self.init_kwargs.get('ignore_hydrogen', False)
        #-----------------------------------------------------------------------
        self.op_rotate = np.eye(3)
        self.op_translate = np.zeros(3)
        self.op_indices = None
        self._rotmat = None
        self._atom_naos = None
        #-----------------------------------------------------------------------
        if not isinstance(atoms, Atoms):
            try:
                atoms = io.read(atoms)
            except Exception as e:
                raise e

        atoms_init = atoms
        atoms_change = None
        if refatoms is not None :
            if hasattr(refatoms, 'atoms'):
                refatoms = refatoms.atoms
            if not isinstance(refatoms, Atoms):
                try:
                    refatoms = io.read(refatoms)
                except Exception as e:
                    raise e
            if refatoms is not atoms :
                self.op_rotate, self.op_translate, self.op_indices = minimize_rmsd_operation(refatoms, atoms,
                        stereo = stereo, rotate_method = rotate_method, reorder_method = reorder_method,
                        use_reflection = use_reflection, ignore_hydrogen = ignore_hydrogen)
                atoms = atoms[self.op_indices]
                atoms.set_positions(np.dot(atoms.positions,self.op_rotate)+self.op_translate)
                atoms_change = atoms
        if atoms_change is None :
            atoms = atoms.copy()
        else :
            atoms = atoms_change
        self.atoms_init = atoms_init
        # self.op_rotate_inv = np.linalg.inv(self.op_rotate)
        self.op_rotate_inv = np.ascontiguousarray(self.op_rotate.T)
        if self.op_indices is None : self.op_indices = np.arange(len(atoms))
        self.op_indices_inv = np.argsort(self.op_indices)
        #-----------------------------------------------------------------------
        if engine is None :
            if engine_name == 'pyscf' :
                from qmlearn.drivers.pyscf import EnginePyscf
                engine_options['mol'] = atoms
                engine_options['method'] = method
                engine_options['basis'] = basis
                engine_options['charge'] = charge
                if isinstance(xc, (str, type(None))) :
                    engine_options['xc'] = xc
                elif isinstance(xc, (list, tuple, set)):
                    engine_options['xc'] = ','.join(xc)
                else :
                    raise AttributeError(f"Not support this '{xc}'")
                engine = EnginePyscf(**engine_options)
            elif engine_name == 'psi4' :
                from qmlearn.drivers.psi4 import EnginePsi4
                engine_options['mol'] = atoms
                engine_options['method'] = method
                engine_options['basis'] = basis
                if isinstance(xc, (str, type(None))) :
                    engine_options['xc'] = xc
                else :
                    raise AttributeError(f"Not support this '{xc}'")
                engine = EnginePsi4(**engine_options)
            else :
                raise AttributeError(f"Sorry, not support '{engine_name}' now")
        #-----------------------------------------------------------------------
        self.refatoms = refatoms or atoms
        self.engine = engine
        self.occs = occs
        self.atoms = atoms
        self.engine_name = engine_name
        self.engine_options = engine_options
        self.method = method
        self.xc = xc
        self.basis = basis
        #-----------------------------------------------------------------------
        return self

    def duplicate(self, atoms, **kwargs):
        r""" Function to create a duplicate image of Atomic coordinates ('refatoms')

        Parameters
        ----------
        atoms : :obj: PySCF or ASE atom object
            Molecular geometry

        Returns
        -------
        obj : :obj: PySCF or ASE atom object
            Reference atoms. """
        for k, v in self.init_kwargs.items():
            if k == 'self' or k in kwargs :
                continue
            elif k == 'atoms' :
                kwargs[k] = atoms
            else :
                kwargs[k] = v
        if kwargs['refatoms'] is None :
            kwargs['refatoms'] = self.atoms
        obj = self.__class__(**kwargs)
        return obj

    def run(self, **kwargs):
        r"""Function to run the External Calculator."""
        self.engine.run(**kwargs)

    @property
    def rotmat(self):
        r""" Rotated density matrix """
        if self._rotmat is None :
            self._rotmat = self.rotation2rotmat(self.op_rotate)
        return self._rotmat

    @property
    def atom_naos(self):
        r""" Number of atomic orbitals. """

        if self._atom_naos is None :
            self._atom_naos = self.get_atom_naos()
        return self._atom_naos

    def convert_back(self, y, prop = 'gamma', rotate = True, reorder = True, **kwargs):
        r""" Function to rotate gamma or forces base on initial coordinates.

        Parameters
        ----------
        pro : str
            Options

            | 1-RDM : 'gamma'
            | Forces : 'forces'

        y : ndarray
            Predicted gamma or forces matrix to be rotated.

        Returns
        -------
        y : ndarray
           Rotated gamma or forces matrix.

        """
        nao = self.nao
        #
        if np.allclose(self.op_rotate, self.op_rotate_inv) : rotate = False
        if np.all(self.op_indices[:-1] < self.op_indices[1:]) : reorder = False
        #
        # print('rotate', rotate, reorder)
        if 'gamma' in prop :
            if y.ndim == 1 : y = y.reshape((nao, nao))
            if rotate : y = self.rotmat.T@y@self.rotmat
            if reorder :
                naos = self.atom_naos
                blocks = array2blocks(y, sections = naos)
                y = blocks2array(blocks, indices = self.op_indices_inv)
        elif 'forces' in prop :
            if y.ndim == 1 : y = y.reshape((-1, 3))
            if rotate : y = np.dot(y, self.op_rotate_inv)
            if reorder :
                y = y[self.op_indices_inv]
        elif 'dipole' == prop :
            if rotate : y = np.dot(y, self.op_rotate_inv)
        elif 'quadrupole' == prop :
            if rotate :
                raise AttributeError(f"{prop} not support rotation.")
        else :
            # others just return it self
            pass
        return y
