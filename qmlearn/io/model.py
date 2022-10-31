from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from qmlearn.model.model import QMModel
from qmlearn.io import read_db
from qmlearn.utils import tenumerate

def db2qmmodel(filename, names = '*', mmodels = None, qmmol_options = None, purify_gamma = True):
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
    if isinstance(filename, dict):
        data = filename
    else :
        data = read_db(filename, names=names)
    refqmmol = data['qmmol']
    if qmmol_options :
        refqmmol.init_kwargs.update(qmmol_options)
        refqmmol = refqmmol.__class__(**refqmmol.init_kwargs)
        print("New options of QMMOL :\n", refqmmol.init_kwargs)
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
    model = QMModel(mmodels=mmodels, refqmmol = refqmmol, purify_gamma = purify_gamma)
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
        print('Start predicting...', flush = True)
        shape = y[0].shape
        if 'gamma_pp' in properties:
            gammas = properties['gamma_pp']
        else :
            gammas = []
            for i, a in tenumerate(train_atoms):
                gamma = model.predict(a, refatoms=a).reshape(shape)
                if model.purify_gamma :
                    gamma = model.qmmol.purify_gamma(gamma)
                gammas.append(gamma)
            properties['gamma_pp'] = gammas
        y = gammas
        print('Start delta learning...', flush = True)
        for k in mmodels :
            if not k.startswith('d_') : continue
            key = k[2:]
            if key not in properties :
                print(f"!WARN : '{key}' not in the database", flush = True)
            model.fit(y, properties[key], method = k)
    print('Finish the reading.', flush = True)
    return model
