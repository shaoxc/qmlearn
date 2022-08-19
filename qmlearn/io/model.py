from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from qmlearn.model.model import QMModel
from qmlearn.io import read_db

def db2qmmodel(filename, names = '*', mmodels = None):
    r"""Train QMModel to learn :math:`{\gamma}` in terms of :math:`V_{ext}` from training data
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
    data = read_db(filename, names=names)
    refqmmol = data['qmmol']
    train_atoms = data['atoms']
    properties = data['properties']
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
    for k in mmodels :
        if k.startswith('d_'):
            delta_learn = True
            break
    else :
        delta_learn = False
    #
    if delta_learn :
        shape = y[0].shape
        gammas = []
        for i, a in enumerate(train_atoms):
            gamma = model.predict(a, refatoms=a).reshape(shape)
            #
            gamma = model.qmmol.purify_gamma(gamma)
            #
            gammas.append(gamma)
        y = gammas
        for k in mmodels :
            if not k.startswith('d_') : continue
            key = k[2:]
            if key not in properties :
                print(f"!WARN : '{key}' not in the database", flush = True)
            model.fit(y, properties[key], method = k)
    return model
