from qmlearn.drivers.core import atoms_rmsd, atoms2bestplane
from qmlearn.io import read_images

def get_train_atoms(traj=None, nsamples=10, skip = 0, tol=0.02, direction = None, refatoms = None):
    images = read_images(traj)
    nsteps = len(images)
    print('nsteps', nsteps, traj)
    j = skip - 1
    data=[]
    for i in range(nsamples):
        k=j+1
        for j in range(k, nsteps):
            atoms = images[j]
            new = True
            rmsd=0.0
            transform = True
            if len(data) == 0 :
                if direction is not None :
                    atoms = atoms2bestplane(atoms, direction = direction)
                elif refatoms is not None :
                        _, atoms = atoms_rmsd(refatoms, atoms, transform=transform)
            else :
                for a in data:
                    rmsd, atoms = atoms_rmsd(a, atoms, transform=transform)
                    transform = False
                    if rmsd < tol :
                        new=False
                        break
            if new:
                data.append(atoms)
                break
    i = len(data)
    if i < nsamples :
        print(f"WARN : Only get {len(data)} samples at {j} step. Maybe you can reduce the 'tol'.", flush = True)
    else :
        print(f'Get {len(data)} samples at {j} step.', flush = True)
    return data
