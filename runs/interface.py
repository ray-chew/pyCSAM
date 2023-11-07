from src import fourier, lin_reg, physics, reconstruction
from src import utils, var, io
import numpy as np

class get_pmf(object):

    def __init__(self, nhi, nhj, U, V, debug=False):
        self.fobj = fourier.f_trans(nhi, nhj)

        self.U = U
        self.V = V

        self.debug = debug

    def sappx(self, cell, lmbda=0.1, summed=False, updt_analysis=False, scale=1.0, refine=False, iter_solve=False):
        self.fobj.do_full(cell)

        if iter_solve:
            am, data_recons = lin_reg.do_iter(self.fobj, cell, lmbda)    
        else:
            am, data_recons = lin_reg.do(self.fobj, cell, lmbda)

        self.fobj.get_freq_grid(am)
        freqs = scale * np.abs(self.fobj.ampls)

        if refine:
            cell.topo_m -= data_recons
            am, data_recons = lin_reg.do(self.fobj, cell, lmbda)

            self.fobj.get_freq_grid(am)
            freqs += scale * np.abs(self.fobj.ampls)

        if self.debug: print("data_recons: ", data_recons.min(), data_recons.max())

        dat_2D = reconstruction.recon_2D(data_recons, cell)

        if self.debug: print("dat_2D: ", dat_2D.min(), dat_2D.max())
        
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

        wlat = np.diff(cell.lat).mean()
        wlon = np.diff(cell.lon).mean()

        kks = np.fft.rfftfreq((ampls.shape[1] * 2) - 1, d=1.0)
        lls = np.fft.fftfreq((ampls.shape[0]), d=1.0)

        ampls = np.fft.fftshift(ampls, axes=0)
        lls = np.fft.fftshift(lls, axes=0)

        kkg, llg = np.meshgrid(kks, lls)

        dat_2D = np.fft.irfft2(np.fft.ifftshift(ampls, axes=0) * ampls.size, s=cell.topo.shape).real

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

        return ampls, uw_pmf_freqs, dat_2D, [kks, lls]


    def cg_spsp(self, cell, freqs, kklls, dat_2D, summed=False, updt_analysis=False, scale=1.0):
        self.fobj.do_cg_spsp(cell)

        self.fobj.m_i = kklls[0]
        self.fobj.m_j = kklls[1]

        freqs = scale * np.abs(freqs)
        
        analysis = var.analysis()
        analysis.get_attrs(self.fobj, freqs)
        analysis.recon = dat_2D

        if updt_analysis: cell.analysis = analysis

        ideal = physics.ideal_pmf(U=self.U, V=self.V)
        uw_pmf_freqs = ideal.compute_uw_pmf(analysis, summed=summed)

        return freqs, uw_pmf_freqs, dat_2D
    


def taper_quad(params, simplex_lat, simplex_lon, cell, topo):
    # get quadrilateral mask
    utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell, topo, rect=True)

    # get tapered mask with padding
    taper = utils.taper(cell, params.padding, art_it=params.taper_art_it)
    taper.do_tapering()    

    # get tapered topography in quadrilateral with padding
    utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell, topo, rect=True, padding=params.padding, topo_mask=taper.p)

def taper_nonquad(params, simplex_lat, simplex_lon, cell, topo):
    # get tapered mask with padding
    taper = utils.taper(cell, params.padding, art_it=params.taper_art_it)
    taper.do_tapering()

    # get padded topography
    utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell, topo, rect=True, padding=params.padding)

    # get padded topography in non-quad
    utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell, topo, rect=False, padding=params.padding, filtered=False)
    # mask_taper = np.copy(cell.mask)

    # apply tapering mask to padded non-quad domain
    utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell, topo, rect=False, padding=params.padding, topo_mask=taper.p, filtered=False, mask=(taper.p > 1e-2).astype(bool))

    # mask=(taper.p > 1e-2).astype(bool)
    # cell.topo = taper.p * cell.topo * mask
    # cell.mask = mask 

    