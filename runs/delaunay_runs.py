# %%
import sys
import os
# set system path to find local modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from src import io, var, utils, physics, delaunay
from wrappers import interface, diagnostics
from vis import plotter, cart_plot

import time

from IPython import get_ipython
ipython = get_ipython()

if '__IPYTHON__' in globals():
    ipython.run_line_magic('load_ext autoreload')
    ipython.run_line_magic('autoreload')


# %%
from inputs.lam_run import params
# from inputs.selected_run_dfft import params
# from inputs.debug_run import params
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
reader.read_dat(params.fn_grid, grid)
grid.apply_f(utils.rad2deg) 

# writer object
writer = io.writer(params.output_fn, params.rect_set, debug=params.debug_writer)

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


# %%
if params.run_full_land_model:
    params.rect_set = delaunay.get_land_cells(tri, topo, height_tol=0.5)
    print(params.rect_set)

params_orig = deepcopy(params)
writer.write_all_attrs(params)
writer.populate('decomposition', 'rect_set', params.rect_set)

# %%
# Plot the loaded topography...
# cart_plot.lat_lon(topo, int=1)

levels = np.linspace(-500.0, 3500.0, 9)
cart_plot.lat_lon_delaunay(topo, tri, levels, label_idxs=True, fs=(12,7), highlight_indices=params.rect_set, output_fig=True, fn='../manuscript/delaunay.pdf', int=1, raster=True)

# %%
# del topo.lat_grid
# del topo.lon_grid

# %%

nhi = params.nhi
nhj = params.nhj

fa = interface.first_appx(nhi, nhj, params, topo)
sa = interface.second_appx(nhi, nhj, params, topo, tri)
# diagnostics object
diag = diagnostics.delaunay_metrics(params, tri, writer=writer)
dplot = diagnostics.diag_plotter(params, nhi, nhj)

if not params.no_corrections:
    rel_errs_orig = []

start = time.time()

for rect_idx in params.rect_set:

    #################################################
    #
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

    if params.debug_writer:
        writer.populate(rect_idx, 'topo_ref', cell_ref.topo)
        writer.populate(rect_idx, 'spectrum_ref', ampls_ref)
        writer.populate(rect_idx, 'pmf_ref', uw_ref)

    v_extent = [fft_2D_ref.min(), fft_2D_ref.max()]
    sols = (cell_ref, ampls_ref, uw_ref, fft_2D_ref)
    dplot.show((rect_idx, rect_idx+1), sols, kls=kls_ref, v_extent = v_extent, dfft_plot=True)

    ###################################
    #
    # Do first approximation
    # 
    if params.dfft_first_guess:
        nhi = len(cell_ref.lon)
        nhj = len(cell_ref.lat)
        sa.nhi = nhi
        sa.nhj = nhj

        ampls_fa, uw_fa, dat_2D_fa, kls_fa = np.copy(ampls_ref), np.copy(uw_ref), np.copy(fft_2D_ref), np.copy(kls_ref)

        cell_fa = cell_ref
    else:
        cell_fa, ampls_fa, uw_fa, dat_2D_fa = fa.do(simplex_lat, simplex_lon)

    diag.update_quad(rect_idx, uw_ref, uw_fa)

    if params.debug_writer:
        writer.populate(rect_idx, 'spectrum_fg', ampls_fa)
        writer.populate(rect_idx, 'recon_fg', dat_2D_fa)
        writer.populate(rect_idx, 'pmf_fg', uw_fa)

    if hasattr(params, "ir_plot_titles"):
        ir_args = [
            "reference FFT reconstruction",
            "FA LSFF power spectrum",
            "FA LSFF PMF spectrum",
            None,
            None
        ]
        sols = (cell_fa, ampls_fa, uw_fa, fft_2D_ref)
        fn = "plots_FA_LSFF"
    else:
        sols = (cell_fa, ampls_fa, uw_fa, dat_2D_fa)
        ir_args, fn = None, None
    
    dfft_plot = True if params.dfft_first_guess else False
    kls = kls=kls_ref if params.dfft_first_guess else None
    dplot.show((rect_idx, rect_idx+1), sols, v_extent=v_extent, ir_args=ir_args, fn=fn, dfft_plot=dfft_plot, kls=kls)

    ###################################
    #
    # Do second approximation over non-
    # quadrilateral grid cells
    # 
    triangle_pair = np.zeros(2, dtype='object')

    for cnt, idx in enumerate(range(rect_idx, rect_idx+2)):
        
        cell, ampls_sa, uw_sa, dat_2D_sa = sa.do(idx, ampls_fa)

        sols = (cell, ampls_sa, uw_sa, dat_2D_sa)
        dplot.show(idx, sols, v_extent=v_extent)

        cell.uw = uw_sa
        triangle_pair[cnt] = cell

        writer.write_all(idx, cell, cell.analysis)
        writer.populate(idx, 'pmf_sg', uw_sa)
        del cell

    ###################################
    #
    # Do iterative refinement?
    # 
    ref_topo = np.copy(cell_ref.topo)
    topo_sum = np.zeros_like(ref_topo)
    rel_err = diag.get_rel_err(triangle_pair)
    if not params.no_corrections: 
        rel_errs_orig.append(rel_err)
        v_extent_orig = np.copy(v_extent)



        if hasattr(params, "ir_plot_titles"):
            freqs_vext = [ampls_fa.min(), ampls_fa.max()]
            pmf_vext = [uw_fa.min(), uw_fa.max()]
            ir_args = [
                "abs. diff in first FA recon.",
                "first combined power spectrum",
                "first combined PMF spectrum",
                freqs_vext,
                pmf_vext
            ]

            first_diff = np.abs(fft_2D_ref - dat_2D_fa)

            sols = (cell_ref, triangle_pair[0].analysis.ampls + triangle_pair[1].analysis.ampls, triangle_pair[0].uw + triangle_pair[1].uw, first_diff)
            dplot.show((idx-1, idx), sols, v_extent=[0,first_diff.max()], ir_args=ir_args, fn="first_plots")

    print(rel_err)
    print(diag)
    corrected = False
    
    ir_cnt = 0
    while np.abs(rel_err) > 0.2 and (not params.no_corrections): 
        mode = "overestimation" if np.sign(rel_err) > 0 else "underestimation"
        print("correcting %s... with n_modes = %i" %(mode, sa.n_modes))

        refinement_pair = np.zeros(2, dtype='object')

        topo_sum += dat_2D_fa
        res_topo = -np.sign(rel_err) * (ref_topo - topo_sum)
        res_topo -= res_topo.mean()

        cell_fa, ampls_fa, uw_fa, dat_2D_fa = fa.do(simplex_lat, simplex_lon, res_topo=res_topo)

        v_extent = [dat_2D_fa.min(), dat_2D_fa.max()]

        # sols = (cell_fa, ampls_fa, uw_fa, dat_2D_fa)
        # dplot.show(idx, sols, v_extent=v_extent)

        for cnt, idx in enumerate(range(rect_idx, rect_idx+2)):
            cell, ampls_rf, uw_rf, dat_2D_rf = sa.do(idx, ampls_fa, res_topo = res_topo)

            ampls_sum = triangle_pair[cnt].analysis.ampls - np.sign(rel_err) * ampls_rf

            cutoff = np.sort(ampls_sum.ravel())[::-1][params.n_modes-1]
            ampls_sum[np.where(ampls_sum < cutoff)] = 0.0

            print((ampls_sum > 0.0).sum())

            cell.analysis.ampls = ampls_sum
            triangle_pair[cnt].analysis.ampls = ampls_sum

            ideal = physics.ideal_pmf(U=params.U, V=params.V)
            uw_pmf_refined = ideal.compute_uw_pmf(cell.analysis, summed=False)

            print("uw_pmf_refined", uw_pmf_refined.sum())

            cell.uw = uw_pmf_refined
            refinement_pair[cnt] = cell

            # sols = (cell, ampls_rf, uw_rf, dat_2D_rf)
            # dplot.show(idx, sols, v_extent=v_extent)

        ir_cnt += 1

        corrected = True
        rel_err = diag.get_rel_err(refinement_pair)
        print(rel_err)
        print(diag)
        # topo_tmp = refinement_pair[0].analysis.recon + refinement_pair[1].analysis.recon

    print(ir_cnt)
    sa.n_modes = params.n_modes

    if corrected:
        triangle_pair = refinement_pair
        
        topo_sum += dat_2D_fa
        final_diff = fft_2D_ref - topo_sum
        sols = (cell, triangle_pair[0].analysis.ampls + triangle_pair[1].analysis.ampls, triangle_pair[0].uw + triangle_pair[1].uw, final_diff)

        if hasattr(params, "ir_plot_titles"):
            ir_args = [
                "abs. diff in final FA recon.",
                "final combined power spectrum",
                "final combined PMF spectrum",
                freqs_vext,
                pmf_vext
            ]
            fn = "final_plots"
            idx = (idx-1, idx)
            vext = [0,first_diff.max()]
        else:
            ir_args = None
            fn = None
            idx = idx
            vext = v_extent_orig

        dplot.show(idx, sols, v_extent=vext, ir_args=ir_args, fn=fn)

    diag.update_pair(triangle_pair)

end = time.time()
# %%
diag.end(verbose=True)
print("time taken = %.2f" %(end-start))

# %%
# print(rel_errs_orig)
print(diag.rel_errs)
plotter.error_bar_plot(params.rect_set, diag.rel_errs, params, gen_title=True)
# plotter.error_bar_plot(params.rect_set, np.abs(fft_rel_errs) - np.abs(diag.rel_errs), params, fs=(14,5), ylim=[-10,10], title="| FFT LRE | - | LSFF LRE |", output_fig=True, fn='../manuscript/dfft_vs_lsff.pdf', fontsize=12)
# plotter.error_bar_plot(params.rect_set, diag.rel_errs, params, gen_title=False, ylabel="", fs=(14,5), ylim=[-100,100], output_fig=True, title="percentage LRE", fn='../manuscript/lre_bar.pdf', fontsize=12, comparison=np.array(rel_errs_orig)*100)
# plotter.error_bar_plot(params.rect_set, diag.rel_errs, params, gen_title=False, ylabel="", fs=(14,5), ylim=[-100,100], output_fig=True, title="percentage LRE", fn='../manuscript/lre_bar.pdf', fontsize=12)

# %%
print(diag.max_errs)
plotter.error_bar_plot(params.rect_set, diag.max_errs, params, gen_title=False, ylabel="", fs=(14,5), ylim=[-100,100], output_fig=True, title="percentage MRE", fontsize=12)

# plotter.error_bar_plot(params.rect_set, diag.max_errs, params, gen_title=False, ylabel="", fs=(14,5), ylim=[-100,100], output_fig=True, title="percentage MRE", fn='../manuscript/mre_bar.pdf', fontsize=12)


# %%
errors = np.zeros((len(tri.simplices)))
errors[:] = np.nan
errors[params.rect_set] = diag.max_errs
errors[np.array(params.rect_set)+1] = diag.max_errs

levels = np.linspace(-1000.0, 3000.0, 5)
cart_plot.error_delaunay(topo, tri, label_idxs=False, fs=(12,8), highlight_indices=params.rect_set, output_fig=True, fn='../manuscript/error_delaunay_fine.pdf', iint=1, errors=errors, alpha_max=0.6)

# %%
print(np.abs(np.array(rel_errs_orig) * 100 ).mean())
np.abs(diag.rel_errs).mean()

# %%
print(np.linalg.norm(final_diff - fft_2D_ref) / np.linalg.norm(fft_2D_ref))
print(np.linalg.norm(first_diff - fft_2D_ref) / np.linalg.norm(fft_2D_ref))
# %%
# time taken = 67.91
fft_rel_errs = np.copy(diag.rel_errs)
fft_max_errs = np.copy(diag.max_errs)
# %%