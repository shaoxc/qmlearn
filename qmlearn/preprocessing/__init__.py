import numpy as np
from qmlearn.drivers.core import atoms_rmsd, atoms2bestplane
from qmlearn.io import read_images

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
