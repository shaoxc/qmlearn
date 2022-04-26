import numpy as np
from ase import Atoms
import scipy.linalg as sla
# from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge

from qmlearn.drivers.mol import QMMol


class QMModel(object):
    def __init__(self, mmodels = None, method='gamma', ncharge=None, nspin = 1, occs = None, refqmmol = None, **kwargs):
        self._method = method
        self.mmodels = mmodels or {}
        #
        if self.method not in self.mmodels :
            self.mmodels[self.method] = KernelRidge(alpha=0.1,kernel='linear')
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
        self._refqmmol = refqmmol
        self.qmmol = None

    @property
    def model(self):
        return self.mmodels[self.method]

    @model.setter
    def model(self, value):
        self.mmodels[self.method] = value

    @property
    def refqmmol(self):
        if self._refqmmol is None :
            raise AttributeError("Please set the 'refqmmol' before the predict.")
        return self._refqmmol

    @refqmmol.setter
    def refqmmol(self, value):
        self._refqmmol = value

    @property
    def method(self):
        if self._method.lower() != 'gamma' :
            raise AttributeError("Only support 'gamma' method now.")
        return self._method

    def fit(self, X, y, model = None, method = None):
        if model is None :
            method = method or self.method
            model = self.mmodels[method]
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

    def predict(self, x, model = None, method = None, **kwargs):
        #
        x = [self.translate_input(x, **kwargs).ravel()]
        if model is None :
            method = method or self.method
            model = self.mmodels[method]
        #
        y = model.predict(x)[0]
        #
        return y

    def convert_back(self, y, prop = 'gamma', qmmol = None, **kwargs):
        qmmol = qmmol or self.qmmol
        y = qmmol.convert_back(y, prop = prop, **kwargs)
        return y

    def translate_input(self, x, **kwargs):
        if isinstance(x, np.ndarray) :
            out = x
        elif isinstance(x, Atoms) :
            x = self.refqmmol.duplicate(x, **kwargs)
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
        w,vr=sla.eig(s_new)
        w=1./np.real(np.sqrt(w))
        s_moh=vr@np.diag(w)@vr.T
        oorb = np.einsum('ij,mj->mi',s_moh,orb)
        s_new = np.einsum('mi,nj,mn->ij', oorb, oorb, s)
        return oorb


class MQMModel(QMModel):
    def __init__(self, mmodels = None, method='gamma', ncharge=None, nspin = 1, occs = None, refqmmol = None,
            fragments = None, qmmodels = None, **kwargs):
        super().__init__(mmodels=mmodels, method=method, ncharge=ncharge, nspin=nspin, occs=occs, refqmmol=refqmmol, **kwargs)
        self._fragments = fragments
        self._qmmodels = qmmodels

    @property
    def qmmodels(self):
        if self._qmmodels is None :
            raise AttributeError("Please set the 'qmmodels' before the predict.")
        return self._qmmodels

    @qmmodels.setter
    def qmmodels(self, value):
        self._qmmodels = value

    @property
    def fragments(self):
        if self._fragments is None :
            raise AttributeError("Please set the 'fragments' before the predict.")
        return self._fragments

    @fragments.setter
    def fragments(self, value):
        self._fragments = value

    def translate_input(self, x, convert = True, **kwargs):
        if isinstance(x, np.ndarray) :
            out = x
        elif isinstance(x, Atoms) :
            x = self.refqmmol.duplicate(x, **kwargs)
        elif isinstance(x, QMMol):
            pass
        else :
            raise AttributeError("Please check the input.")

        if isinstance(x, QMMol):
            self.qmmol = x
            out = self.qmmol.vext
            block = self.get_block_vext(x, convert = convert)
            out = out - sla.block_diag(*block)
        return out

    def get_block_vext(self, qmmol, convert = True, **kwargs):
        if isinstance(qmmol, Atoms) :
            qmmol = self.refqmmol.duplicate(qmmol, **kwargs)
        self.sub_qmmols = []
        block = []
        for i, index in enumerate(self.fragments):
            a = self.qmmodels[i].refqmmol.duplicate(qmmol.atoms[index], **kwargs)
            self.sub_qmmols.append(a)
            if convert :
                v = a.convert_back(a.vext, prop = 'gamma')
            else :
                v = a.vext
            block.append(v)
        return block

    def predict(self, x, method = None, convert=True, split=True, **kwargs):
        x = self.translate_input(x, convert = convert, **kwargs).ravel()
        method = method or self.method
        model = self.mmodels[method]
        y = model.predict([x])[0]
        y = np.asarray(y)
        block = self.predict_block(method=method, convert=convert)
        if 'gamma' in method :
            y_frags = sla.block_diag(*block)
        elif 'force' in method :
            y_frags = np.vstack(block)
        elif y.size == 1 :
            y_frags = np.sum(block)
        else :
            raise AttributeError("'MQMModel' only for 'energy', 'force' and 'gamma' now.")
        if y.size > 1 : y = y.reshape(y_frags.shape)

        if split:
            return y_frags, y
        else:
            return y + y_frags

    def predict_block(self, x=None, method=None, convert =True, **kwargs):
        if x is not None :
            # This one just to make sure the 'sub_qmmols' were calculated.
            x = self.translate_input(x, convert = convert, **kwargs).ravel()
        method = method or self.method
        block = []
        for i, mol in enumerate(self.sub_qmmols) :
            if method == 'gamma' :
                x0 = mol.vext
            else :
                x0 = mol.gamma
            y0 = self.qmmodels[i].predict(x0, method=method)
            if np.size(y0) == np.size(x0) :
                y0 = y0.reshape(x0.shape)
                if method == 'gamma' : # save the gamma
                    mol.gamma = y0
            if convert :
                y0 = mol.convert_back(y0, prop = method)
            block.append(y0)
        return block
