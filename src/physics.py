import numpy as np


class ideal_pmf(object):

    def __init__(self, **kwarg):
        self.N = 0.02       # reference brunt-väisälä frequnecy [s^{-1}]
        self.U = -10.0      # reference horizontal wind [m s^{-1}]
        self.V = 2.0        # reference vertical wind [m s^{-1}]
        self.AE = 6371.0008 * 1E3 # Earth's radius in [m]

        # If keyword arguments are specified, we use those values...
        for key, value in kwarg.items():
            setattr(self, key, value)

    def compute_uw_pmf(self, analysis, summed=True):
        N = self.N
        U = self.U
        V = self.V

        wlat = analysis.wlat
        wlon = analysis.wlon

        kks = analysis.kks * 2.0 * np.pi
        lls = analysis.lls * 2.0 * np.pi

        # if ((kks.ndim == 1) and (lls.ndim == 1)):
        #     print(True)
        #     ampls = analysis.ampls[np.nonzero(analysis.ampls)]
        # else:
        #     ampls = analysis.ampls
        ampls = analysis.ampls

        wla = wlat #* self.AE
        wlo = wlon #* self.AE

        kks = kks / wlo
        lls = lls / wla

        om = -kks * U -lls * V
        omsq = om**2

        mms = (N**2 * (kks**2 + lls**2) / omsq) - (kks**2 + lls**2)
        ampls[np.where(mms <= 0.0)] = 0.0
        mms[np.isnan(mms)] = 0.0
        mms = np.sqrt(mms)

        # wave-action density
        Ag = - 0.5 * ((ampls)**2 * N**2 / om )
        Ag[np.isinf(Ag)] = 0.0
        Ag[np.isnan(Ag)] = 0.0

        # group velocity in z-direction
        cgz = self.N * (kks**2 + lls**2)**0.5 * mms / (kks**2 + lls**2 + mms**2)**(3/2)

        cgz[np.isnan(cgz)] = 0.0

        uw_pmf = (Ag * kks * cgz)

        if summed:
            return uw_pmf.sum()
        else:
            return uw_pmf
        

        