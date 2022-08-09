import numpy as np
import itertools as it

class Reflection(object):
    def __init__(self, **kwargs):
        self._srot = None
        self._srot_stereo = None
        self._srot_stereo_no = None

    @property
    def srot(self):
        if self._srot is None :
            self.build_srot()
        return self._srot

    @property
    def srot_stereo(self):
        if self._srot_stereo is None :
            self.build_srot()
        return self._srot_stereo

    @property
    def srot_stereo_no(self):
        if self._srot_stereo_no is None :
            self.build_srot()
        return self._srot_stereo_no

    def build_srot(self):
        srot=np.zeros((48,3,3))
        srot_stereo = []
        srot_stereo_no = []
        mr = np.array(list(it.product([1,-1], repeat=3)))
        i=0
        for swap in it.permutations(range(3)):
            for ijk in mr:
                srot[i][tuple([(0,1,2),swap])]= ijk
                i+=1
        for rot in srot :
            if np.linalg.det(rot) > 0.0 :
                srot_stereo.append(rot)
            else :
                srot_stereo_no.append(rot)
        self._srot = srot
        self._srot_stereo = np.asarray(srot_stereo)
        self._srot_stereo_no = np.asarray(srot_stereo_no)


REFLECTION = Reflection()
