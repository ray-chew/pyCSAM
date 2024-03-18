import numpy as np


class f_trans(object):
    """
    Fourier transformer class
    """

    def __init__(self, nhar_i, nhar_j):
        """
        Initalises a discrete spectral space with the corresponding Fourier coefficients spanning ``nhar_i`` and ``nhar_j``.

        Parameters
        ----------
        nhar_i : int
            number of spectral modes in the first horizontal direction
        nhar_j : int
            number of spectral modes in the second horizontal direction
        """
        self.nhar_i = nhar_i
        self.nhar_j = nhar_j

        self.m_i = None
        self.m_j = None

        self.pick_kls = False
        self.components = "imag"

    def __get_IJ(self, cell):
        """
        Private method to compute :math:`x / \Delta x`.
        """
        if self.grad:
            lon, lat = cell.grad_lon, cell.grad_lat
            lon_m, lat_m = cell.grad_lon_m, cell.grad_lat_m
        else:
            lon, lat = cell.lon, cell.lat
            lon_m, lat_m = cell.lon_m, cell.lat_m

        # now define appropriate indices for the points withing the triangle
        # by shifting the origin to the minimum lat and lon
        lat_res = np.diff(lat).mean()
        lon_res = np.diff(lon).mean()

        self.wlat = cell.wlat
        self.wlon = cell.wlon

        lat_res = cell.wlat
        lon_res = cell.wlon

        self.J = np.ceil((lat_m - lat_m.min()) / lat_res).astype(int)
        self.I = np.ceil((lon_m - lon_m.min()) / lon_res).astype(int)

    def __prepare_terms(self, cell):
        """
        Private method that defines the terms comprising the Fourier coefficients
        """
        if self.grad:
            lon_m, lat_m = cell.grad_lon_m, cell.grad_lat_m
        else:
            lon_m, lat_m = cell.lon_m, cell.lat_m

        self.Ni, self.Nj = np.unique(lon_m).size, np.unique(lat_m).size

        self.m_i = np.arange(0, self.nhar_i)

        if self.nhar_j == 2:
            self.m_j = np.arange(-self.nhar_j / 2 + 1, self.nhar_j / 2 + 1)
        elif self.nhar_j % 2 == 0:
            # if self.components == 'real':
            #     self.m_j = np.arange(0, self.nhar_j)
            # else:
            self.m_j = np.arange(-self.nhar_j / 2 + 1, self.nhar_j / 2 + 1)
        else:
            # if self.components == 'real':
            #     self.m_j = np.arange(0, self.nhar_j)
            # else:
            self.m_j = np.arange(-(self.nhar_j - 1) / 2, (self.nhar_j + 1) / 2)

        self.term1 = self.m_i.reshape(1, -1) * self.I.reshape(-1, 1) / self.Ni
        self.term2 = self.m_j.reshape(1, -1) * self.J.reshape(-1, 1) / self.Nj

    def set_kls(self, k_rng, l_rng, recompute_nhij=True, components="imag"):
        """
        Method to select a smaller subset of the dense spectral space, e.g., in the Second Approximation step of the algorithm if the First Approximation is computed with a fast-Fourier transform.

        Parameters
        ----------
        k_rng : list
            list containing the selected k-wavenumber indices
        l_rng : list
            list containing the selected k-wavenumber indices
        recompute_nhij : bool, optional
            resets ``nhar_i`` and ``nhar_j``, by default True
        components : str, optional
            `real` recomputes the spectral space comprising only real spectral components, by default 'imag'
        """
        self.k_idx = np.array(k_rng).astype(int)
        self.l_idx = np.array(l_rng).astype(int)

        k_max = max(self.k_idx)

        if recompute_nhij:
            if k_max % 2 == 1:
                k_max += 1

            # l_max = max(self.l_idx)
            self.nhar_i = int(max(k_max + 1, 2))
            # self.nhar_j = int(max((2.0*l_max),2))

            if components == "real":
                self.components = "real"
                l_max = max(self.l_idx)
                if l_max % 2 == 1:
                    l_max += 1
                # self.nhar_j = int(max(l_max+1,2))

        self.pick_kls = True

    def do_full(self, cell, grad=False):
        r"""
        Assembles the sine and cosine terms that make up the Fourier coefficients in the ``M`` matrix required in the :func:`linear regression <src.lin_reg.do>` computation:

        .. math:: M a_m =h

        Parameters
        ----------
        cell : :class:`src.var.topo_cell` instance
            cell object instance
        grad : bool, optional
            deprecated argument, by default False
        """
        self.typ = "full"

        if grad is True:
            self.grad = True
        else:
            self.grad = False
        self.__get_IJ(cell)
        self.__prepare_terms(cell)

        self.term1 = np.expand_dims(self.term1, -1)
        self.term1 = np.repeat(self.term1, self.nhar_j, -1)
        self.term2 = np.expand_dims(self.term2, 1)
        self.term2 = np.repeat(self.term2, self.nhar_i, 1)

        tt_sum = self.term1 + self.term2

        del self.term1
        del self.term2

        if self.pick_kls:
            tt_sum = tt_sum[:, self.k_idx, self.l_idx]
        else:
            tt_sum = tt_sum.reshape(tt_sum.shape[0], -1)

        bcos = np.cos(2.0 * np.pi * (tt_sum))
        bsin = np.sin(2.0 * np.pi * (tt_sum))

        del tt_sum

        if (self.nhar_i == 2) and (self.nhar_j == 2) and (self.pick_kls == False):
            Ncos = bcos[:, :]
            Nsin = bsin[:, 1:]

        elif self.pick_kls == True:
            Ncos = bcos
            Nsin = bsin

        else:
            if self.nhar_j % 2 == 0:
                Ncos = bcos[:, int(self.nhar_j / 2 - 1) :]
                Nsin = bsin[:, int(self.nhar_j / 2) :]
            else:
                Ncos = bcos[:, int(self.nhar_j / 2 - 1) :]
                Nsin = bsin[:, int(self.nhar_j / 2) :]
            # Ncos = bcos
            # Nsin = np.delete(bsin, int(self.nhar_j/2)-1, axis=1)

        self.bf_cos = Ncos
        self.bf_sin = Nsin
        self.nc = self.bf_cos.shape[1]

    def do_axial(self, cell, alpha=0.0):
        """
        Computes spectral modes along the ``(k,l)``-axes.

        .. deprecated:: 0.90.0

        """
        self.typ = "axial"
        self.__get_IJ(cell)
        self.__prepare_terms(cell)

        alpha = alpha / 180.0 * np.pi

        ktil = self.m_i * np.cos(alpha)
        ltil = self.m_i * np.sin(alpha)

        self.term1 = (
            ktil.reshape(1, -1) * self.I.reshape(-1, 1) / self.Ni
            + ltil.reshape(1, -1) * self.J.reshape(-1, 1) / self.Nj
        )

        khat = self.m_j * np.cos(alpha + np.pi / 2.0)
        lhat = self.m_j * np.sin(alpha + np.pi / 2.0)

        self.term2 = (
            khat.reshape(1, -1) * self.I.reshape(-1, 1) / self.Ni
            + lhat.reshape(1, -1) * self.J.reshape(-1, 1) / self.Nj
        )

        bcos = 2.0 * np.cos(
            2.0 * np.pi * np.hstack([self.term1, self.term2[:, int(self.nhar_j / 2) :]])
        )
        bsin = 2.0 * np.sin(
            2.0
            * np.pi
            * np.hstack([self.term1[:, 1:], self.term2[:, int(self.nhar_j / 2) :]])
        )

        self.bf_cos = bcos
        self.bf_sin = bsin
        self.nc = self.bf_cos.shape[1]

    def do_cg_spsp(self, cell):
        """
        Computes the coarse-grained sparse spectral space

        .. deprecated:: 0.90.0

        """
        self.typ = "full"
        self.grad = False

        self.__get_IJ(cell)
        self.__prepare_terms(cell)

    def get_freq_grid(self, a_m):
        """
        Assembles a dense representation of the sparse spectral space given the Fourier amplitudes computed in the linear regression step.

        Parameters
        ----------
        a_m : list
            list of (sparse) Fourier amplitudes
        """
        nhar_i, nhar_j = self.nhar_i, self.nhar_j

        fourier_coeff = np.zeros((nhar_i, nhar_j))
        nc = self.nc

        zrs = np.zeros((int(self.nhar_j / 2) - 1))
        zrs[:] = np.nan
        # zrs = []

        if (self.typ == "full") and (not self.pick_kls):
            cos_terms = a_m[:nc]
            sin_terms = a_m[nc:]

            if (nhar_i == 2) and (nhar_j == 2):
                sin_terms = np.concatenate(([0.0], sin_terms))

            elif (nhar_i > 2) and (nhar_j > 2):
                cos_terms = np.concatenate((zrs, cos_terms))
                sin_terms = np.concatenate((zrs, [0.0], sin_terms))

            fourier_coeff = cos_terms + 1.0j * sin_terms  # / 2.0
            fourier_coeff = fourier_coeff.reshape(nhar_i, nhar_j).swapaxes(1, 0)

        if (self.typ == "full") and (self.pick_kls):
            cos_terms = a_m[: len(self.k_idx)]
            sin_terms = a_m[len(self.k_idx) :]

            fourier_coeff = np.zeros((nhar_i, nhar_j), dtype=np.complex_)

            for cnt, (row, col) in enumerate(zip(self.k_idx, self.l_idx)):
                fourier_coeff[row, col] = cos_terms[cnt] + 1.0j * sin_terms[cnt]
            fourier_coeff = fourier_coeff.reshape(nhar_i, nhar_j).swapaxes(1, 0)

        if self.typ == "axial":
            f00 = a_m[0]
            cos_terms = a_m[:nc]
            sin_terms = a_m[nc:]
            sin_terms = np.concatenate(([0.0], sin_terms))

            if nhar_j % 2 == 0:
                k_terms = cos_terms[:nhar_i] + 1.0j * sin_terms[:nhar_i]  # / 2.0
                l_terms = cos_terms[nhar_i:] + 1.0j * sin_terms[nhar_i:]  # / 2.0

                l_blk = np.zeros((int(nhar_j / 2 - 1), int(nhar_i)))
                u_blk = np.zeros((int(nhar_j / 2), int(nhar_i - 1)))

                u_blk = np.hstack((l_terms.reshape(-1, 1), u_blk))

                fourier_coeff = np.vstack((l_blk, k_terms, u_blk))

            else:
                y_axs = (
                    cos_terms[: int((nhar_j + 1) / 2 + 1)]
                    + 1.0j * sin_terms[: int((nhar_j + 1) / 2 + 1)]
                )  # / 2.0
                x_axs = (
                    cos_terms[int((nhar_j - 1) / 2) :]
                    + 1.0j * sin_terms[int((nhar_j - 1) / 2) :]
                )  # / 2.0
                x_axs = x_axs.reshape(-1, 1)
                l_blk = np.zeros((int(nhar_i - 1), int((nhar_j - 1) / 2 - 1)))
                u_blk = np.zeros((int(nhar_i - 1), int((nhar_j - 1) / 2)))

                r1 = np.hstack(([0] * int(nhar_j / 2), [f00], y_axs)).reshape(1, -1)
                r2 = np.hstack((u_blk, x_axs, l_blk))
                fourier_coeff = np.vstack((r1, r2))
                fourier_coeff = fourier_coeff.T

        self.ampls = fourier_coeff
