import os
from ase import Atoms, io

def read_images(traj, format=None, nsteps=None):
    if not format :
        format = os.path.splitext(traj)[-1][1:]
    if format : format = format.lower()

    inds = slice(nsteps)

    if isinstance(traj[0], Atoms):
        images = traj[inds]
    elif format == 'traj' :
        from ase.io.trajectory import Trajectory
        images = Trajectory(traj)
        if nsteps : images = images[inds]
    elif format in ['xyz', 'exyz', 'extxyz'] :
        images = io.read(traj, index = inds)[inds]
    else :
        raise AttributeError(f"Sorry, not support '{format}' format.")
    return images
