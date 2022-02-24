from ase.io.trajectory import Trajectory

from qmlearn.drivers.core import atoms_rmsd, atoms2bestplane

def get_train_atoms(mdtraj=None, nsamples=10, skip = 500, tol=0.02, direction = None, refatoms = None):
    traj = Trajectory(mdtraj)
    nsteps = len(traj)
    j = skip
    data=[]
    for i in range(nsamples):
        k=j
        for j in range(k, nsteps):
            atoms = traj[j]
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
    traj.close()
    return data
