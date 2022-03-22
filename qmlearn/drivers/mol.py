import numpy as np
from ase import Atoms, io

from qmlearn.drivers.core import minimize_rmsd_operation

qm_engines = {
        'pyscf' : None
        }

class QMMol(object):
    engine_calcs = [
            'calc_gamma',
            'calc_ncharge',
            'calc_etotal',
            'calc_ke',
            'calc_dipole',
            'calc_quadrupole',
            'calc_forces',
            'calc_idempotency',
            'vext',
            'ovlp',
            ]

    def __init__(self, atoms = None, engine_name = 'pyscf', method = 'rks', basis = '6-31g',
            xc = None, occs=None, refatoms = None, engine_options = {}, charge = None, **kwargs):
        # Save all the kwargs for duplicate
        self.init_kwargs = locals()
        self.init()

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
        #-----------------------------------------------------------------------
        self.op_rotate = np.eye(3)
        self.op_translate = np.zeros(3)
        #-----------------------------------------------------------------------
        if not isinstance(atoms, Atoms):
            try:
                atoms = io.read(atoms)
            except Exception as e:
                raise e

        if refatoms is not None :
            if hasattr(refatoms, 'atoms'):
                refatoms = refatoms.atoms
            if not isinstance(refatoms, Atoms):
                try:
                    refatoms = io.read(refatoms)
                except Exception as e:
                    raise e
            if refatoms is not atoms :
                self.op_rotate, self.op_translate = minimize_rmsd_operation(refatoms, atoms)
                atoms.set_positions(np.dot(atoms.positions,self.op_rotate)+self.op_translate)
        self.op_rotate_inv = np.linalg.inv(self.op_rotate)
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
        self.refatoms = refatoms
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
        for k, v in self.init_kwargs.items():
            if k == 'self' :
                continue
            elif k == 'atoms' :
                kwargs[k] = atoms
                continue
            else :
                kwargs[k] = v
        if kwargs['refatoms'] is None :
            kwargs['refatoms'] = self.atoms
        obj = self.__class__(**kwargs)
        return obj

    def run(self, **kwargs):
        self.engine.run(**kwargs)

    def __getattr__(self, attr):
        if attr in dir(self):
            return object.__getattribute__(self, attr)
        elif attr in self.engine_calcs :
            if not hasattr(self.engine, attr):
                raise AttributeError(f"Sorry, the {self.engine_name} engine not support the {attr} now.")
            return getattr(self.engine, attr)
        else :
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'.")
