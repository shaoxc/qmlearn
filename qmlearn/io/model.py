from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from qmlearn.model.model import QMModel
from qmlearn.io.hdf5 import DBHDF5

def db2qmmodel(filename, names = '*', mmodels = None):
    db = DBHDF5(filename)
    if isinstance(names, str):
        prefix = names
        names = dict.fromkeys(['qmmol', 'atoms', 'properties'])
        names['qmmol'] = db.get_names(prefix + '/qmmol*')[0]
        names['atoms'] = db.get_names(prefix + '/train_atoms*')[0]
        names['properties'] = db.get_names(prefix + '/train_prop*')[0]
        print(f'Guess DB name : {names}', flush = True)
    refqmmol = db.read_qmmol(names['qmmol'])
    train_atoms = db.read_images(names['atoms'])
    properties = db.read_properties(names['properties'])
    db.close()
    #
    X = properties['vext']
    y = properties['gamma']
    #
    if mmodels is None :
        mmodels={
            'gamma': KernelRidge(alpha=0.1,kernel='linear'),
            'd_gamma': LinearRegression(),
            'd_energy': LinearRegression(),
            'd_forces': LinearRegression(),
            'd_dipole': LinearRegression(),
        }
        print(f'Guess mmodels: {mmodels}', flush = True)
    model = QMModel(mmodels=mmodels, refqmmol = refqmmol)
    model.fit(X, y)
    #
    if 'd_gamma' in mmodels :
        shape = y[0].shape
        gammas = []
        for i, mol in enumerate(train_atoms):
            # Do not rotate the molecule
            gamma = model.predict(mol, refatoms=mol).reshape(shape)
            gammas.append(gamma)
        model.fit(gammas, properties['gamma'], model=model.mmodels['d_gamma'])
        y = gammas
    for k in mmodels :
        if not k.startswith('d_') or k in ['d_gamma'] : continue
        key = k[2:]
        if key not in properties :
            raise AttributeError(f"{key} not in the database")
        model.fit(y, properties[key], method = k)
    return model
