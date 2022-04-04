import unittest
import numpy as np
import ase
import ase.build

from qmlearn.drivers.mol import QMMol
from qmlearn.drivers.core import minimize_rmsd_operation

class Water(unittest.TestCase):
    def test_qmmol(self):
        atoms = ase.build.molecule('H2O')
        basis = 'cc-pvTZ'
        xc = 'lda,vwn_rpa'
        method = 'rks'
        charge = 0
        qmmol = QMMol(atoms = atoms, method = method, basis=basis, xc = xc, charge=charge)
        qmmol.run()
        etotal = qmmol.engine.etotal
        self.assertAlmostEqual(etotal, -76.09381930374022, 12)

    def test_rmsd_rotation(self):
        from scipy.spatial.transform import Rotation
        atoms = ase.build.molecule('H2O')
        angle = [0.2, -0.3, 0.4]
        rotation = Rotation.from_euler('zyz', angle).as_matrix()
        atoms2 = atoms.copy()
        atoms2.positions[:] = atoms2.positions @ rotation
        rotation2, _ = minimize_rmsd_operation(atoms2, atoms)
        self.assertTrue(np.allclose(rotation, rotation2))

    def test_rotmat(self):
        from scipy.spatial.transform import Rotation
        atoms = ase.build.molecule('H2O')
        angle = [0.2, -0.3, 0.4]
        rotation = Rotation.from_euler('zyz', angle).as_matrix()
        atoms2 = atoms.copy()
        atoms2.positions[:] = atoms2.positions @ rotation
        atoms3 = atoms2.copy()
        rotation2, _ = minimize_rmsd_operation(atoms, atoms2)
        #
        atoms = ase.build.molecule('H2O')
        basis = 'cc-pvTZ'
        xc = 'lda,vwn_rpa'
        method = 'rks'
        #
        qmmol = QMMol(atoms = atoms, method = method, basis=basis, xc = xc)
        s = qmmol.ovlp
        qmmol2 = qmmol.duplicate(atoms2)
        s2 = qmmol2.ovlp
        self.assertTrue(np.allclose(s, s2))
        #
        qmmol3 = qmmol.duplicate(atoms3, refatoms = atoms3)
        s3 = qmmol3.ovlp
        rotmat = qmmol3.rotation2rotmat(qmmol2.op_rotate)
        s2_back = rotmat.T @ s2 @ rotmat
        self.assertTrue(np.allclose(s2_back, s3))


if __name__ == "__main__":
    print("Tests for rotate")
    unittest.main()
