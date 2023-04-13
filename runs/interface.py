from src import fourier, lin_reg, physics, reconstruction
from src import utils, var, io

import numpy as np

class get_pmf(object):

    def __init__(self, nhi, nhj, U, V, debug=False):
        self.fobj = fourier.f_trans(nhi, nhj)

        self.U = U
        self.V = V

        self.debug = debug

    def sappx(self, cell, lmbda=0.1, summed=False, updt_analysis=False, scale=1.0):
        self.fobj.do_full(cell)
        am, data_recons = lin_reg.do(self.fobj, cell, lmbda)

        if self.debug: print("data_recons: ", data_recons.min(), data_recons.max())

        dat_2D = reconstruction.recon_2D(data_recons, cell)

        if self.debug: print("dat_2D: ", dat_2D.min(), dat_2D.max())

        self.fobj.get_freq_grid(am)

        freqs = scale * np.abs(self.fobj.ampls)
        
        analysis = var.analysis()
        analysis.get_attrs(self.fobj, freqs)
        analysis.recon = dat_2D

        if updt_analysis: cell.analysis = analysis

        ideal = physics.ideal_pmf(U=self.U, V=self.V)

        uw_pmf_freqs = ideal.compute_uw_pmf(analysis, summed=summed)

        return freqs, uw_pmf_freqs, dat_2D


    def dfft(self, cell, summed=False, updt_analysis=False):
        ampls = np.fft.rfft2(cell.topo - cell.topo.mean())
        ampls /= ampls.size

        wlat = np.diff(cell.lat).max()
        wlon = np.diff(cell.lon).max()

        kks = np.fft.rfftfreq((ampls.shape[1] * 2) - 1, d=1.0)
        lls = np.fft.fftfreq((ampls.shape[0]), d=1.0)

        ampls = np.fft.fftshift(ampls, axes=0)
        lls = np.fft.fftshift(lls, axes=0)

        kkg, llg = np.meshgrid(kks, lls)

        dat_2D = np.fft.irfft2(np.fft.ifftshift(ampls, axes=0) * ampls.size).real 

        ampls = np.abs(ampls)

        if self.debug: print(np.sort(ampls.reshape(-1,))[::-1][:25])

        analysis = var.analysis()
        analysis.wlat = wlat
        analysis.wlon = wlon
        analysis.ampls = ampls
        analysis.kks = kkg
        analysis.lls = llg
        analysis.recon = dat_2D

        if updt_analysis: cell.analysis = analysis            

        ideal = physics.ideal_pmf(U=self.U, V=self.V)
        uw_pmf_freqs = ideal.compute_uw_pmf(analysis, summed=summed)

        return uw_pmf_freqs, ampls, dat_2D, [kks, lls]


    