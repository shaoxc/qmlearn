import numpy as np
import ase.units as units
from qmlearn.drivers.core import atoms_rmsd, atoms2bestplane
from qmlearn.io import read_images
from sklearn.decomposition import PCA

def get_train_atoms(traj=None, nsamples=None, skip = 0, tol=0.01, direction = None, transform = True, refatoms = None):
    images = read_images(traj)
    nsteps = len(images)
    if nsamples is None : nsamples = nsteps
    # print('nsteps', nsteps, traj)
    j = skip - 1
    data=[]
    if direction is not None :
        # Because the reference atoms already changed
        transform = True
    for i in range(nsamples):
        k=j+1
        for j in range(k, nsteps):
            atoms = images[j]
            new = True
            if not transform :
                atoms.positions[:] = atoms.positions - np.mean(atoms.positions, axis = 0)
                # atoms.positions[:] = atoms.positions - atoms.get_center_of_mass()
            if len(data) == 0 :
                if direction is not None :
                    atoms = atoms2bestplane(atoms, direction = direction)
                elif refatoms is not None :
                        _, atoms = atoms_rmsd(refatoms, atoms, transform=transform)
            else :
                for ia, a in enumerate(data):
                    convert = transform if ia == 0 else False
                    rmsd, atoms = atoms_rmsd(a, atoms, transform=convert)
                    if rmsd < tol :
                        new=False
                        break
            if new:
                data.append(atoms)
                break
    j += 1
    i = len(data)
    if i < nsamples :
        print(f"WARN : Only get {len(data)} samples at {j} step. Maybe you can reduce the 'tol'.", flush = True)
    else :
        print(f'Get {len(data)} samples at {j} step.', flush = True)
    return data


class AtomsCreater(object):
    def __init__(self, vibrations = None, temperature = 300, random_seed = None, maximum = 1E8):
        self.temperature = temperature*units.kB
        self.vibrations = vibrations
        self.maximum = maximum
        self.random_seed = random_seed
        #
        if hasattr(self.vibrations, 'get_vibrations'):
            self.vibrations = self.vibrations.get_vibrations()
        self._initial()

    def _initial(self):
        vibrations = self.vibrations
        #
        atoms = vibrations.get_atoms().copy()
        pos = PCA().fit_transform(atoms.positions)
        if np.all(pos[:, 1] < 1E-8): # linear molecule
            nstart = 5
        else : # nonlinear molecule
            nstart = 6
        #
        indices = np.arange(nstart, 3*len(atoms))
        modes = vibrations.get_modes(all_atoms=True)[indices].copy()
        energies = np.abs(vibrations.get_energies())[indices].copy()

        for i in range(len(modes)):
            modes[i] /= np.sqrt(np.abs(energies[i]))
        #
        self.modes = modes
        self.atoms = atoms
        #
        self.random = np.random.default_rng(self.random_seed)
        d = len(self.modes)
        self.sigma = np.sqrt(self.temperature/(d*(1-2/(9*d))**3))

    def get_new_atoms(self, sigma = None):
        sigma = sigma or self.sigma
        d = len(self.modes)
        atoms = self.atoms.copy()
        coef = self.random.normal(0, sigma, size = (d, 1))[:,0]
        for c,m in zip(coef, self.modes):
            atoms.positions += c*m
        return atoms

    def _get_new_atoms_v1(self):
        atoms = self.atoms.copy()
        for i, m in enumerate(self.modes):
            c = self.randoms.normal(0, self.sigma)
            atoms.positions += c*m
        return atoms

    def _get_new_atoms_v0(self):
        atoms = self.atoms.copy()
        modes = self.modes
        coef = np.random.random((1, len(modes)))[0]
        coef -= 0.5
        for c,m in zip(coef, modes):
            atoms.positions += c*m
        return atoms

    def __iter__(self):
        self._iter = 0
        return self

    def __next__(self):
        if self._iter <= self.maximum:
            self._iter += 1
            return self.get_new_atoms()
        else:
            raise IndexError(f'Reaches the maximum : {self.maximum}')

def _build_train_atoms(images, data = [], nsamples = 30, tol = 0.01, **kwargs):
    data = data or []
    for istep, atoms in enumerate(images):
        print('istep', istep, len(data))
        for ia, a in enumerate(data):
            if ia < 1:
                use_reflection = True
                rotate_method = 'kabsch'
            else:
                use_reflection = False
                rotate_method = 'none'
            rmsd, atoms = atoms_rmsd(a, atoms, use_reflection=use_reflection, rotate_method=rotate_method, rmsd_cut = tol, **kwargs)
            if rmsd < tol :
                break
        else :
            data.append(atoms)
            if len(data) == nsamples: break
    print(f'Get {len(data)} samples at {istep + 1} step.', flush = True)
    return data

def build_train_atoms(images, nsamples = 30, tol = 0.01, maxtry = 20, **kwargs):
    images = iter(images)
    data=[next(images)]

    def _check_images(data, tol = 0.01):
        poss=[]
        for atoms in data:
            poss.append(atoms.positions)
        poss = np.asarray(poss)
        #
        refatoms = atoms.copy()
        refatoms.positions[:] = poss.mean(axis=0)
        new = [refatoms]
        for atoms in data:
            rmsd, atoms = atoms_rmsd(refatoms, atoms, **kwargs)
            if rmsd < tol : continue
            new.append(atoms)
        if len(new) > len(data): new = new[:len(data)]
        return new

    for i in range(maxtry):
        data = _build_train_atoms(images, data, nsamples = nsamples, tol = tol, **kwargs)
        data = _check_images(data, tol)
        if len(data) == nsamples:
            break
    else :
        ns = len(data)
        raise AttributeError(f"Only found {ns} samples, try increase `maxtry` or change the inputs.")
    return data
