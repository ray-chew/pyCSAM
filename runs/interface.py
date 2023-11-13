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
    


def taper_quad(params, simplex_lat, simplex_lon, cell, topo, res_topo=None):
    # get quadrilateral mask
    utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell, topo, rect=True)

    # get tapered mask with padding
    taper = utils.taper(cell, params.padding, art_it=params.taper_art_it)
    taper.do_tapering()

    # get tapered topography in quadrilateral with padding
    utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell, topo, rect=True, padding=params.padding, topo_mask=taper.p)

def taper_nonquad(params, simplex_lat, simplex_lon, cell, topo, res_topo=None):
    # get tapered mask with padding
    taper = utils.taper(cell, params.padding, art_it=params.taper_art_it)
    taper.do_tapering()

    # get padded topography
    utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell, topo, rect=True, padding=params.padding)
    
    if res_topo is not None:
        cell.topo = res_topo

    # get padded topography in non-quad
    utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell, topo, rect=False, padding=params.padding, filtered=False)
    # mask_taper = np.copy(cell.mask)

    # apply tapering mask to padded non-quad domain
    utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell, topo, rect=False, padding=params.padding, topo_mask=taper.p, filtered=False, mask=(taper.p > 1e-2).astype(bool))

    # mask=(taper.p > 1e-2).astype(bool)
    # cell.topo = taper.p * cell.topo * mask
    # cell.mask = mask 


class first_appx(object):

    def __init__(self, nhi, nhj, params, topo):
        self.nhi, self.nhj = nhi, nhj
        self.params = params
        self.topo = topo

    def do(self, simplex_lat, simplex_lon, res_topo=None):
        cell_fa = var.topo_cell()

        if res_topo is None:
            if self.params.taper_fa:
                taper_quad(self.params, simplex_lat, simplex_lon, cell_fa, self.topo, res_topo)
            else:
                utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell_fa, self.topo, rect=self.params.rect)    
        else:
            cell_fa.topo = res_topo
            utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell_fa, self.topo, padding=self.params.padding, rect=False, mask=np.ones_like(res_topo).astype(bool))    

        first_guess = get_pmf(self.nhi,self.nhj,self.params.U,self.params.V)

        ampls_fa, uw_fa, dat_2D_fa = first_guess.sappx(cell_fa, lmbda=self.params.lmbda_fa, iter_solve=self.params.fa_iter_solve)
        return cell_fa, ampls_fa, uw_fa, dat_2D_fa

class second_appx(object):

    def __init__(self, nhi,nhj, params, topo, tri):
        self.params = params
        self.topo = topo
        self.tri = tri
        self.nhi, self.nhj = nhi, nhj
        self.n_modes = params.n_modes

    def do(self, idx, ampls_fa, res_topo=None):

        # make a copy of the spectrum obtained from the FA.
        fq_cpy = np.copy(ampls_fa)
        fq_cpy[np.isnan(fq_cpy)] = 0.0 # necessary. Otherwise, popping with fq_cpy.max() gives the np.nan entries first.

        cell = var.topo_cell()

        simplex_lat = self.tri.tri_lat_verts[idx]
        simplex_lon = self.tri.tri_lon_verts[idx]

        # use the non-quadrilateral self.topography
        utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell, self.topo, rect=True)

        if (res_topo is not None) and (not self.params.taper_sa):
            cell.topo = res_topo * cell.mask

        utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell, self.topo, rect=False, filtered=False)

        if self.params.taper_sa:
            taper_nonquad(self.params, simplex_lat, simplex_lon, cell, self.topo, res_topo=res_topo)

        second_guess = get_pmf(self.nhi,self.nhj,self.params.U,self.params.V)

        indices = []
        modes_cnt = 0
        while modes_cnt < self.n_modes:
            max_idx = np.unravel_index(fq_cpy.argmax(), fq_cpy.shape)
            # skip the k = 0 column
            # if max_idx[1] == 0:
            #     fq_cpy[max_idx] = 0.0
            # # else we want to use them
            # else:
            indices.append(max_idx)
            fq_cpy[max_idx] = 0.0
            modes_cnt += 1

        if not self.params.cg_spsp:
            k_idxs = [pair[1] for pair in indices]
            l_idxs = [pair[0] for pair in indices]

        if self.params.dfft_first_guess:
            second_guess.fobj.set_kls(k_idxs, l_idxs, recompute_nhij = True, components='real')
        else:
            second_guess.fobj.set_kls(k_idxs, l_idxs, recompute_nhij = False)

        ampls_sa, uw_sa, dat_2D_sa = second_guess.sappx(cell, lmbda=self.params.lmbda_sa, updt_analysis=True, scale=1.0, iter_solve=self.params.sa_iter_solve)      
        
        return cell, ampls_sa, uw_sa, dat_2D_sa