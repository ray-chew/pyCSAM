# %%
import sys
import os
# set system path to find local modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import importlib
import matplotlib.pyplot as plt

from src import io, var, utils, fourier, physics, delaunay
from runs import interface
from vis import plotter, cart_plot

%load_ext autoreload
%autoreload

# %%
# from runs.lam_run import params
from runs.selected_run_dfft import params
# from runs.debug_run import params
from copy import deepcopy

# print run parameters, for sanity check.
if params.self_test():
    params.print()

# %%
# initialise data objects
grid = var.grid()
topo = var.topo_cell()

# read grid
reader = io.ncdata(padding=params.padding, padding_tol=(60-params.padding))

# writer object
writer = io.writer(params.output_fn, params.rect_set, debug=params.debug_writer)

reader.read_dat(params.fn_grid, grid)
grid.apply_f(utils.rad2deg) 

# we only keep the topography that is inside this lat-lon extent.
lat_verts = np.array(params.lat_extent)
lon_verts = np.array(params.lon_extent)

# read topography
if not params.enable_merit:
    reader.read_dat(params.fn_topo, topo)
    reader.read_topo(topo, topo, lon_verts, lat_verts)
else:
    reader.read_merit_topo(topo, params)
    topo.topo[np.where(topo.topo < -500.0)] = -500.0

topo.gen_mgrids()

tri = delaunay.get_decomposition(topo, xnp=params.delaunay_xnp, ynp=params.delaunay_ynp, padding = reader.padding)
writer.write_all('decomposition', tri)
writer.populate('decomposition', 'rect_set', params.rect_set)

# %%
if params.run_full_land_model:
    params.rect_set = delaunay.get_land_cells(tri, topo, height_tol=0.5)
    print(params.rect_set)

params_orig = deepcopy(params)
writer.write_all_attrs(params)
# %%
# Plot the loaded topography...
%autoreload
# cart_plot.lat_lon(topo, int=1)

levels = np.linspace(-500.0, 3500.0, 9)
cart_plot.lat_lon_delaunay(topo, tri, levels, label_idxs=True, fs=(20,12), highlight_indices=params.rect_set, output_fig=True, fn='../manuscript/delaunay.pdf', int=1, raster=True)

# %%
# del topo.lat_grid
# del topo.lon_grid

# %%
%autoreload
pmf_diff = []
pmf_refs = []
pmf_sums = []
pmf_fas  = []
pmf_ssums= []
idx_name = []

nhi = params.nhi
nhj = params.nhj

for rect_idx in params.rect_set:

    #################################################
    # compute DFFT over reference quadrilateral cell.
    #
    print("computing reference quadrilateral cell: ", (rect_idx, rect_idx+1))

    cell_ref = var.topo_cell()
    
    simplex_lat = tri.tri_lat_verts[rect_idx]
    simplex_lon = tri.tri_lon_verts[rect_idx]

    if params.taper_ref:
        interface.taper_quad(params, simplex_lat, simplex_lon, cell_ref, topo)
    else:
        utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell_ref, topo, rect=params.rect)    

    ref_run = interface.get_pmf(nhi,nhj,params.U,params.V)
    ampls_ref, uw_ref, fft_2D_ref, kls_ref = ref_run.dfft(cell_ref)

    v_extent = [fft_2D_ref.min(), fft_2D_ref.max()]

    if params.plot:
        fs = (15,5.0)
        fig, axs = plt.subplots(1,3, figsize=fs)
        fig_obj = plotter.fig_obj(fig, nhi, nhj)
        axs[0] = fig_obj.phys_panel(axs[0], fft_2D_ref, title='T%i + T%i: Reference FFT reconstruction' %(rect_idx, rect_idx+1), xlabel='longitude [km]', ylabel='latitude [km]', extent=[cell_ref.lon.min(), cell_ref.lon.max(), cell_ref.lat.min(), cell_ref.lat.max()], v_extent=v_extent)

        axs[1] = fig_obj.fft_freq_panel(axs[1], ampls_ref, kls_ref[0], kls_ref[1], typ='real')
        axs[2] = fig_obj.fft_freq_panel(axs[2], uw_ref, kls_ref[0], kls_ref[1], title="FFT PMF spectrum", typ='real')
        plt.tight_layout()
        plt.show()


    ###################################
    #
    # Do first approximation
    # 
    if params.dfft_first_guess:
        nhi = len(cell_ref.lon)
        nhj = len(cell_ref.lat)

        ampls_fa, uw_fa, dat_2D_fa, kls_fa = np.copy(ampls_ref), np.copy(uw_ref), np.copy(fft_2D_ref), np.copy(kls_ref)

        cell_fa = cell_ref
    else:
        cell_fa = var.topo_cell()

        if params.taper_fa:
            interface.taper_quad(params, simplex_lat, simplex_lon, cell_fa, topo)
        else:
            utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell_fa, topo, rect=params.rect)    

        first_guess = interface.get_pmf(nhi,nhj,params.U,params.V)

        ampls_fa, uw_fa, dat_2D_fa = first_guess.sappx(cell_fa, lmbda=params.lmbda_fa, iter_solve=params.fa_iter_solve)

    if params.plot:
        fs = (15.0,4.0)
        fig, axs = plt.subplots(1,3, figsize=fs)
        fig_obj = plotter.fig_obj(fig, nhi, nhj)
        axs[0] = fig_obj.phys_panel(axs[0], dat_2D_fa, title='T%i+T%i: FF reconstruction' %(rect_idx,rect_idx+1), xlabel='longitude [km]', ylabel='latitude [km]', extent=[cell_fa.lon.min(), cell_fa.lon.max(), cell_fa.lat.min(), cell_fa.lat.max()], v_extent=v_extent)

        if params.dfft_first_guess:
            axs[1] = fig_obj.fft_freq_panel(axs[1], ampls_fa, kls_fa[0], kls_fa[1], typ='real')
            axs[2] = fig_obj.fft_freq_panel(axs[2], uw_fa, kls_fa[0], kls_fa[1], title="PMF spectrum", typ='real')
        else:
            axs[1] = fig_obj.freq_panel(axs[1], ampls_fa)
            axs[2] = fig_obj.freq_panel(axs[2], uw_fa, title="PMF spectrum")

        plt.tight_layout()
        plt.show()
    
    triangle_pair = np.zeros(2, dtype='object')
    for cnt, idx in enumerate(range(rect_idx, rect_idx+2)):
        # make a copy of the spectrum obtained from the FA.
        fq_cpy = np.copy(ampls_fa)
        fq_cpy[np.isnan(fq_cpy)] = 0.0 # necessary. Otherwise, popping with fq_cpy.max() gives the np.nan entries first.

        cell = var.topo_cell()

        print("computing idex: ", idx)

        simplex_lat = tri.tri_lat_verts[idx]
        simplex_lon = tri.tri_lon_verts[idx]

        # use the non-quadrilateral topography
        utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell, topo, rect=True)
        utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell, topo, rect=False, filtered=False)

        if params.taper_sa:
            interface.taper_nonquad(params, simplex_lat, simplex_lon, cell, topo)
        # else:

        #     utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell, topo, rect=False, filtered=False)

        second_guess = interface.get_pmf(nhi,nhj,params.U,params.V)

        indices = []
        modes_cnt = 0
        while modes_cnt < params.n_modes:
            max_idx = np.unravel_index(fq_cpy.argmax(), fq_cpy.shape)
            # skip the k = 0 column
            # if max_idx[1] == 0:
            #     fq_cpy[max_idx] = 0.0
            # # else we want to use them
            # else:
            indices.append(max_idx)
            fq_cpy[max_idx] = 0.0
            modes_cnt += 1

        if not params.cg_spsp:
            k_idxs = [pair[1] for pair in indices]
            l_idxs = [pair[0] for pair in indices]

        if params.dfft_first_guess:
            second_guess.fobj.set_kls(k_idxs, l_idxs, recompute_nhij = True, components='real')
        else:
            second_guess.fobj.set_kls(k_idxs, l_idxs, recompute_nhij = False)

        ampls_sa, uw_sa, dat_2D_sa = second_guess.sappx(cell, lmbda=params.lmbda_sa, updt_analysis=True, scale=1.0, iter_solve=params.sa_iter_solve)      

        if params.plot:
            fs = (15,4.0)
            fig, axs = plt.subplots(1,3, figsize=fs)
            fig_obj = plotter.fig_obj(fig, nhi, nhj)
            axs[0] = fig_obj.phys_panel(axs[0], dat_2D_sa, title='T%i: Reconstruction' %idx, xlabel='longitude [km]', ylabel='latitude [km]', extent=[cell.lon.min(), cell.lon.max(), cell.lat.min(), cell.lat.max()], v_extent=v_extent)
            if params.dfft_first_guess:
                axs[1] = fig_obj.fft_freq_panel(axs[1], ampls_sa, kls_fa[0], kls_fa[1], typ='real')
                axs[2] = fig_obj.fft_freq_panel(axs[2], uw_sa, kls_fa[0], kls_fa[1], title="PMF spectrum", typ='real')
            else:
                axs[1] = fig_obj.freq_panel(axs[1], ampls_sa)
                axs[2] = fig_obj.freq_panel(axs[2], uw_sa, title="PMF spectrum")
            plt.tight_layout()
            # plt.savefig('../output/T%i.pdf' %idx)
            plt.show()

        cell.uw = uw_sa
        triangle_pair[cnt] = cell
        del cell

    uw_0 = triangle_pair[0].uw.sum()
    uw_1 = triangle_pair[1].uw.sum()
    uw_sum = uw_0 + uw_1

    ampls_0 = triangle_pair[0].analysis.ampls
    ampls_1 = triangle_pair[1].analysis.ampls
    ampls_sum = (ampls_0 + ampls_1)
    analysis_sum = triangle_pair[0].analysis
    analysis_sum.ampls = ampls_sum

    ideal = physics.ideal_pmf(U=params.U, V=params.V)
    uw_spec_sum = 0.5 * ideal.compute_uw_pmf(analysis_sum)

    print(uw_0, uw_1)
    print(uw_ref.sum(), uw_fa.sum(), uw_sum, uw_spec_sum)

    pmf_refs.append(uw_ref.sum())
    pmf_fas.append(uw_fa.sum())
    pmf_sums.append(uw_sum)
    pmf_ssums.append(uw_spec_sum)

writer.populate('decomposition', 'pmf_diff', pmf_diff)

# %%
def get_rel_diff(arr, ref):
    arr = np.array(arr)
    ref = np.array(ref)

    return arr / ref - 1.0


def get_max_diff(arr, ref, max):
    arr = np.array(arr)
    ref = np.array(ref)

    return (arr - ref) / max

sum_diff = get_max_diff(pmf_sums, pmf_refs, np.array(pmf_refs).max())
sum_diff = get_rel_diff(pmf_sums, pmf_refs)


pmf_percent_diff = np.array(sum_diff) * 100
plotter.error_bar_plot(params.rect_set, pmf_percent_diff, params, gen_title=True)



# %%
importlib.reload(io)
importlib.reload(cart_plot)


errors = np.zeros((len(tri.simplices)))
errors[:] = np.nan
errors[params.rect_set] = pmf_percent_diff
errors[np.array(params.rect_set)+1] = pmf_percent_diff

levels = np.linspace(-1000.0, 3000.0, 5)
cart_plot.error_delaunay(topo, tri, label_idxs=True, fs=(12,8), highlight_indices=params.rect_set, output_fig=False, iint=1, errors=errors, alpha_max=0.6)

# %%
