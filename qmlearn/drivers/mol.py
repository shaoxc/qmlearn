# import numpy as np
from ase import Atoms, io

from qmlearn.drivers.core import atoms_rmsd

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
            'calc_idempotency'
            ]

    def __init__(self, atoms = None, engine_name = 'pyscf', method = 'rks', basis = '6-31g',
            xc = None, occs=None, refatoms = None, engine_options = {}, **kwargs):
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
                self.rmsd, atoms = atoms_rmsd(refatoms, atoms, keep = False)
            else :
                self.rmsd = 0.0
        else :
            self.rmsd = None
        #-----------------------------------------------------------------------
        if engine is None :
            if engine_name == 'pyscf' :
                from qmlearn.drivers.pyscf import EnginePyscf
                engine_options['mol'] = atoms
                engine_options['method'] = method
                engine_options['basis'] = basis
                if isinstance(xc, str) :
                    engine_options['xc'] = xc
                elif isinstance(xc, (list, tuple, set)):
                    engine_options['xc'] = ','.join(xc)
                engine = EnginePyscf(**engine_options)
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

    @property
    def ncharge0(self):
        return self.engine.ncharge0

    @property
    def vext(self):
        return self.engine.vext

    @property
    def ovlp(self):
        return self.engine.ovlp

    def __getattr__(self, attr):
        if attr in dir(self):
            return object.__getattribute__(self, attr)
        elif attr in self.engine_calcs :
            if not hasattr(self.engine, attr):
                raise AttributeError(f"Sorry, the {self.engine_name} engine not support the {attr} now.")
            return getattr(self.engine, attr)
        else :
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'.")
