import numpy as np
from ase import Atoms
from scipy.linalg import eig
# from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge

from qmlearn.drivers.mol import QMMol


class QMModel(object):
    def __init__(self, model = None, mmodels = None, method='gamma', ncharge=None, nspin = 1, occs = None, refqmmol = None, **kwargs):
        self._method = method
        self.mmodels = mmodels or {}
        if self.method not in self.mmodels :
            self.mmodels[self.method] = model
        #
        if self.model is None :
            self.model = KernelRidge(alpha=0.1,kernel='linear')
        # for orbital
        self.nspin = nspin
        self.ncharge = ncharge
        self.occs = occs
        if self.occs is None and self.ncharge :
            if self.nspin == 1 :
                self.occs = np.ones(self.ncharge//2)*2
            elif self.nspin == 2 :
                self.occs = np.ones(self.ncharge)
        #
        self.refqmmol = refqmmol
        self.qmmol = None

    @property
    def model(self):
        return self.mmodels[self.method]

    @model.setter
    def model(self, value):
        self.mmodels[self.method] = value

    @property
    def method(self):
        return self._method

    def fit(self, X, y, model = None):
        if model is None : model = self.model
        #
        if isinstance(X[0], np.ndarray):
            if X[0].ndim > 1 :
                X = [item.ravel() for item in X]
        if isinstance(y[0], np.ndarray):
            if y[0].ndim > 1 :
                y = [item.ravel() for item in y]
        #
        model.fit(X, y)
        return model

    def predict(self, x, model = None, nsamples=1):
        #
        if nsamples == 1 :
            x = [self.translate_input(x).ravel()]
        if model is None : model = self.model
        #
        y = model.predict(x)
        if nsamples == 1 : y = y[0]
        #
        return y

    def translate_input(self, x):
        if isinstance(x, np.ndarray) :
            out = x
        elif isinstance(x, Atoms) :
            if self.refqmmol is None :
                raise AttributeError("Please change the input or set the 'refqmmol' before the predict.")
            x = self.refqmmol.duplicate(x)
        elif isinstance(x, QMMol):
            pass
        else :
            raise AttributeError("Please check the input.")

        if isinstance(x, QMMol):
            self.qmmol = x
            out = x.vext
        return out

    def orth_orb(self, s, orb):
        s_new = np.einsum('mi,nj,mn->ij', orb, orb, s)
        w,vr=eig(s_new)
        w=1./np.real(np.sqrt(w))
        s_moh=vr@np.diag(w)@vr.T
        oorb = np.einsum('ij,mj->mi',s_moh,orb)
        s_new = np.einsum('mi,nj,mn->ij', oorb, oorb, s)
        return oorb
