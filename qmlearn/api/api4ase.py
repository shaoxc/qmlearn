import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Ha, Bohr


class QMLCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'dipole', 'stress']

    def __init__(self, qmmodel = None, second_learn = {}, method = 'gamma',
            label='QMLearn', atoms=None, directory='.', refqmmol = None, **kwargs):
        Calculator.__init__(self, label = label, atoms = atoms, directory = directory, **kwargs)
        self.qmmodel = qmmodel
        self.second_learn = second_learn
        self.method = method
        self._refqmmol = refqmmol

    @property
    def refqmmol(self):
        if self._refqmmol is None :
            if hasattr(self.qmmodel, 'refqmmol'):
                return self.qmmodel.refqmmol
            else :
                # qmmodel is refqmmol
                return self.qmmodel
        return self._refqmmol

    @refqmmol.setter
    def refqmmol(self, value):
        self._refqmmol = value

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        properties=['energy', 'forces']
        Calculator.calculate(self,atoms=atoms,properties=properties,system_changes=system_changes)
        atoms = atoms or self.atoms
        self.results['stress'] = np.zeros(6)
        if self.method == 'engine' :
            qmmol = self.refqmmol.duplicate(atoms, refatoms=atoms)
            self.calc_with_engine(qmmol, properties=properties)
        else :
            if self.method == 'gamma' :
                qmmol = self.refqmmol.duplicate(atoms.copy())
                self.calc_with_gamma(qmmol, properties=properties)
            else :
                raise AttributeError(f"Sorry, not support '{self.method}' now.")

    def calc_with_gamma(self, qmmol, properties = ['energy']):
        shape = self.qmmodel.refqmmol.vext.shape
        gamma = self.qmmodel.predict(qmmol).reshape(shape)
        m2 = self.second_learn.get('gamma', None)
        if m2 :
            gamma2 = self.qmmodel.predict(gamma, method = m2).reshape(shape)
        else :
            gamma2 = gamma

        # if 'energy' in properties :
        if 'energy' :
            m2 = self.second_learn.get('energy', None)
            if m2 :
                energy = self.qmmodel.predict(gamma, method=m2)
                self.results['energy'] = energy * Ha
            else :
                energy = qmmol.calc_etotal(gamma2)
                self.results['energy'] = energy * Ha

        if 'forces' in properties:
            m2 = self.second_learn.get('forces', None)
            if m2 :
                forces = self.qmmodel.predict(gamma, method=m2).reshape((-1, 3))
            else :
                forces = qmmol.calc_forces(gamma2).reshape((-1, 3))
            forces = np.dot(forces, qmmol.op_rotate_inv)
            self.results['forces'] = forces[qmmol.op_indices_inv] * Ha/Bohr
            #
            # forces_shift = np.mean(self.results['forces'], axis = 0)
            # print('Forces shift :', forces_shift, flush = True)
            # self.results['forces'] -= forces_shift
            #

        if True :
        # if 'dipole' in properties :
            m2 = self.second_learn.get('dipole', None)
            if m2 :
                dipole = self.qmmodel.predict(gamma, method=m2)
            else :
                dipole = qmmol.calc_dipole(gamma2)
            dipole = np.dot(dipole, qmmol.op_rotate_inv)
            self.results['dipole'] = dipole * Bohr
        # print('energy', self.results['energy'], flush = True)
        self.results['gamma'] = gamma

    def calc_with_engine(self, qmmol, properties = ['energy']):
        qmmol.engine.run()
        energy = qmmol.engine.etotal
        self.results['energy'] = energy * Ha
        self.results['gamma'] = qmmol.engine.gamma
        if 'forces' in properties:
            forces = qmmol.engine.forces
            print('ff', forces)
            # forces = np.dot(forces, qmmol.op_rotate_inv)
            self.results['forces'] = forces * Ha/Bohr

        if True :
        # if 'dipole' in properties :
            dipole = qmmol.calc_dipole(qmmol.engine.gamma)
            self.results['dipole'] = dipole * Bohr
