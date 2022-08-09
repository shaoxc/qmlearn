from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from qmlearn.model.model import QMModel
from qmlearn.io.hdf5 import DBHDF5

def db2qmmodel(filename, names = '*', mmodels = None):
    """Train QMModel to learn :math:`{\gamma}` in terms of :math:`V_{ext}` from training data
    then an additional layer of training learn :math:`{\delta}E` and :math:`{\delta}{\gamma}` 
    based on previously learned :math:`{\gamma}`.
    
    Parameters
    ----------
    filename : str
        Name of database file
    names : str, optional
        name of database, by default '*'
    mmodels : dict, optional
        set of machine learning models used for training , If not provided
        by default KKR will be used to learn gamma and linear regression for
        delta learning

    Returns
    -------
    model : obj
        trained model
    """
    db = DBHDF5(filename)
    if isinstance(names, str):
        prefix = names
        names = dict.fromkeys(['qmmol', 'atoms', 'properties'])
        names['qmmol'] = db.get_names(prefix + '/qmmol*')[0]
        # names['atoms'] = db.get_names(prefix + '/train_atoms*')[0]
        names['properties'] = db.get_names(prefix + '/train_prop*')[0]
        print(f'Guess DB names : {names}', flush = True)
    refqmmol = db.read_qmmol(names['qmmol'])
    # train_atoms = db.read_images(names['atoms'])
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
        }
        print(f'Guess mmodels: {mmodels}', flush = True)
    model = QMModel(mmodels=mmodels, refqmmol = refqmmol)
    model.fit(X, y)
    #
    if 'd_gamma' in mmodels :
        shape = y[0].shape
        gammas = []
        for i, vext in enumerate(X):
            gamma = model.predict(vext).reshape(shape)
            gammas.append(gamma)
        y = gammas
        model.fit(y, properties['gamma'], method = 'd_gamma')
    for k in mmodels :
        if not k.startswith('d_') or k in ['d_gamma'] : continue
        key = k[2:]
        if key not in properties :
            print(f"!WARN : '{key}' not in the database", flush = True)
        model.fit(y, properties[key], method = k)
    return model
