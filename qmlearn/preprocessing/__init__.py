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
    def __init__(self, modes= None, frequencies=None, energies=None, atoms=None, temperature = 300, random_seed = None, maximum = 1E8):
        self.temperature = temperature*units.kB
        self.modes = np.asarray(modes)
        self.maximum = maximum
        self.atoms = atoms
        self.random_seed = random_seed
        if energies is None:
            if frequencies is None:
                raise ValueError("Please provide the vibrational frequencies or energies")
            energies = frequencies*units.invcm
        self.energies = energies
        #
        self._initial()

    def _initial(self):
        atoms = self.atoms
        #
        pos = PCA().fit_transform(atoms.positions)
        if np.all(pos[:, 1] < 1E-8): # linear molecule
            nstart = 5
        else : # nonlinear molecule
            nstart = 6
        #
        indices = np.arange(nstart, 3*len(atoms))
        d = len(indices)
        if len(self.modes) == 3*len(atoms):
            self.modes = self.modes[indices]
            self.energies = self.energies[indices]
        elif len(self.modes) != d:
            raise ValueError("The wrong number of normal modes.")
        #
        uc = units._hbar * units.m / np.sqrt(units._e * units._amu)
        emode_ev = uc**2
        mid = self.temperature*len(atoms)*2*emode_ev
        chi_mid = d*(1-2/(9*d))**3
        sigma = np.sqrt(mid/chi_mid)
        #
        self.sigma = sigma
        self.random = np.random.default_rng(self.random_seed)
        #

    def get_new_atoms(self, sigma = None):
        sigma = sigma or self.sigma
        d = len(self.modes)
        atoms = self.atoms.copy()
        coef = self.random.normal(0, sigma, size = (d, 1))[:,0]
        coef /= self.energies
        for c,m in zip(coef, self.modes):
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

def _build_train_atoms_v1(images, nsamples = 30, tol = 0.01, maxtry = 20, **kwargs):
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

def build_train_atoms(images, nsamples = 30, tol = 0.01, use_reflection = False, rotate_method = 'kabsch', data = None, **kwargs):
    images = iter(images)
    if data is None : data=[next(images)]

    for istep, atoms in enumerate(images):
        print('istep', istep, len(data))
        for ia, a in enumerate(data):
            rmsd, _ = atoms_rmsd(a, atoms, rmsd_cut = tol, use_reflection = use_reflection, rotate_method = rotate_method, **kwargs)
            if rmsd < tol : break
        else :
            data.append(atoms)
            if len(data) == nsamples: break
    #-----------------------------------------------------------------------
    print('Find a reference atoms :', flush = True)
    diffs = []
    refatoms = data[0]
    for atoms in data[1:] :
        rmsd, _ = atoms_rmsd(refatoms, atoms, rmsd_cut = tol, use_reflection = use_reflection, rotate_method = rotate_method, **kwargs)
        diffs.append(rmsd)
    diffs = np.asarray(diffs)
    index = np.argmax(diffs)+1
    refatoms = data.pop(index)
    new = [refatoms]
    for atoms in data:
        rmsd, atoms = atoms_rmsd(refatoms, atoms, rotate_method = rotate_method, **kwargs)
        new.append(atoms)
    #-----------------------------------------------------------------------
    print(f'Get {len(new)} samples at {istep + 1} step.', flush = True)
    return new
