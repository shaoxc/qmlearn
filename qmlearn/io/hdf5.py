import fnmatch

import numpy as np
from ase import Atoms
import h5py

from qmlearn.drivers.mol import QMMol

class DBHDF5(object):
    def __init__(self, filename, mode = 'a', qmmol = None):
        self.fh = h5py.File(filename, 'a')
        self._qmmol = qmmol
        self._group = None

    @property
    def qmmol(self):
        return self._qmmol

    @qmmol.setter
    def qmmol(self, value):
        self._qmmol = value

    @property
    def group(self):
        return self._group

    @group.setter
    def group(self, value):
        self._group = value

    def get_all_names(self, fh = None, attr = None):
        class H5names:
            def __init__(self, attr=None):
                self.names = []
                self.attr = attr

            def __call__(self, name, obj):
                if self.attr :
                    if hasattr(obj, self.attr) : self.names += [name]
                else :
                    self.names += [name]
        fh = fh or self.fh
        h5names = H5names(attr)
        fh.visititems(h5names)
        return h5names.names

    @property
    def names(self):
        names = self.get_all_names()
        names = [item for item in names if item.count('/') < 2]
        return names

    def get_names(self, name = '*'):
        r"""
        Note :
            Only support '*', '?', '[seq]' and '[!seq]' with fnmatch
        """
        sch = ['*', '?', '[']
        if any(x in name for x in sch):
            name = fnmatch.filter(self.names, name)
        elif isinstance(name, str):
            name = [name]
        return name

    def write_qmmol(self, qmmol = None, name = None, **kwargs):
        qmmol = qmmol or self.qmmol
        if qmmol is None :
            raise AttributeError(f"Please set 'qmmol' for {self.__class__.__name__}.")
        if name is None : name = qmmol.method+'/qmmol'
        if name in self.fh :
            self.group = self.fh[name].parent
            print(f"!WARN : '{name}' already in the database, so do nothing.")
            return
        group = self.fh.create_group(name)
        for k, v in qmmol.init_kwargs.items():
            if k in ['self'] : continue
            if v is None : continue
            if k in ['atoms', 'refatoms'] :
                g = group.create_group(k)
                self._write_atoms(g, v)
            elif hasattr(v, 'keys') :
                g = group.create_group(k)
                self._write_dict(g, v)
            else :
                group[k] = v
        self.group = group.parent

    def read_qmmol(self, name, **kwargs):
        group = self.fh[name]
        dicts = {}
        for k, v in group.items():
            if v is None : continue
            if k in ['atoms', 'refatoms'] :
                dicts[k] = self._read_atoms(v)
            elif hasattr(v, 'keys'):
                dicts[k] = {}
                self._read_dict(v, dicts[k])
            else :
                dicts[k] = self._encode(v)
        qmmol = QMMol(**dicts)
        return qmmol

    def write_properties(self, properties = None, prefix = 'train', name = None, **kwargs):
        if not name :
            if self.group is None : self.write_qmmol(**kwargs)
            for k,v in properties.items():
                if hasattr(v, 'keys') or len(v) == 0 : continue
                nsamples = len(v)
                break
            name = self.group.name + '/' + prefix+'_props_'+str(nsamples)
        if name in self.fh :
            print(f"!WARN : '{name}' already in the database, so do nothing.")
            return
        group = self.fh.create_group(name)
        self._write_dict(group, properties)

    def read_properties(self, name, **kwargs):
        return self._read_dict(self.fh[name])

    def write_images(self, images = None, prefix = 'train', name = None, **kwargs):
        if name is None :
            if self.group is None : self.write_qmmol(**kwargs)
            nsamples = len(images)
            name = self.group.name + '/' + prefix+'_atoms_'+str(nsamples)
        if name in self.fh :
            print(f"!WARN : '{name}' already in the database, so do nothing.")
            return
        group = self.fh.create_group(name)
        for i, atoms in enumerate(images):
            g = group.create_group(str(i))
            self._write_atoms(g, atoms)

    def read_images(self, name = None, **kwargs):
        group = self.fh[name]
        images = []
        ids = []
        for k, v in group.items():
            ids.append(k)
            images.append(self._read_atoms(v))
        inds = np.argsort(np.array(ids, dtype=int))
        images = [images[x] for x in inds]
        return images

    def _write_dict(self, g, d):
        for k, v in d.items() :
            if v is None : continue
            if hasattr(v, 'keys') :
                g1 = g.create_group(k)
                self._write_dict(g1, v)
            elif isinstance(v, (list, tuple)):
                g.create_dataset(k, data=np.asarray(v))
            else :
                g[k] = v

    def _read_dict(self, g, d=None):
        if d is None : d = {}
        for k, v in g.items() :
            if hasattr(v, 'keys') :
                d[k] = {}
                self._read_dict(v, d[k])
            else :
                d[k] = self._encode(v)
        return d

    def _write_atoms(self, g, atoms):
        dt = h5py.string_dtype(encoding='utf-8')
        g["symbols"] = np.array(atoms.get_chemical_symbols(), dtype = dt)
        g["positions"] = atoms.positions
        g["cell"] = atoms.cell

    def _read_atoms(self, g):
        symbols = g["symbols"][()].astype(str)
        positions = g["positions"][()]
        cell = g["cell"][()]
        atoms = Atoms(symbols, positions = positions, cell = cell)
        return atoms

    def _encode(self, v, encoding = 'UTF-8'):
        v = v[()]
        if isinstance(v, bytes):
            v = v.decode('UTF-8')
        elif isinstance(v, np.ndarray) and len(v) > 0 and isinstance(v[0], bytes):
            v = v.astype(str)
        return v

    def close(self):
        self.fh.close()
