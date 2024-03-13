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
from wrappers import interface
from vis import plotter, cart_plot

from IPython import get_ipython
ipython = get_ipython()

if '__IPYTHON__' in globals():
    ipython.run_line_magic('load_ext autoreload')
    ipython.run_line_magic('autoreload')

from sys import exit
if __name__ != "__main__": exit(0)

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
# cart_plot.lat_lon(topo, int=1)

levels = np.linspace(-500.0, 3500.0, 9)
cart_plot.lat_lon_delaunay(topo, tri, levels, label_idxs=True, fs=(10,6), highlight_indices=params.rect_set, output_fig=True, fn='../manuscript/delaunay.pdf', int=1, raster=True)

# %%
# del topo.lat_grid
# del topo.lon_grid

# %%
pmf_diff = []
pmf_sum_diff = []
idx_name = []
for rect_idx in params.rect_set:
    corrected = False
    tried_correction = False
    params = deepcopy(params_orig)
    while not corrected:  # check if I need a deepcopy here
        all_cells = np.zeros(2, dtype='object')
        for cnt, idx in enumerate(range(rect_idx,rect_idx+2)):
            # initialise cell object
            cell = var.topo_cell()

            print("computing idx:", idx)

            simplex_lat = tri.tri_lat_verts[idx]
            simplex_lon = tri.tri_lon_verts[idx]

            if params.tapering:
                fg_rect = True if params.taper_full_fg else False
                utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell, topo, rect=fg_rect)
            
                taper = utils.taper(cell, params.padding, art_it=10)
                taper.do_tapering()

                if params.taper_second or params.taper_both:
                    utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell, topo, rect=False, padding=params.padding)
                    mask_taper = np.copy(cell.mask)
                    utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell, topo, rect=params.rect)
                
                if (params.taper_first) or params.taper_both:
                    utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell, topo, rect=True, padding=params.padding, topo_mask=taper.p)
            else:
                utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell, topo, rect=params.rect)    

            topo_orig = np.copy(cell.topo)
            mask_orig = np.copy(cell.mask)
            
            if params.dfft_first_guess:
                nhi = len(cell.lon)
                nhj = len(cell.lat)
            else:
                nhi = params.nhi
                nhj = params.nhj

            first_guess = interface.get_pmf(nhi,nhj,params.U,params.V)

            #######################################################

            if params.debug:
                print("cell.topo: ", cell.topo.min(), cell.topo.max())
                print("cell.lon: ", cell.lon.min(), cell.lon.max())
                print("cell.lat: ", cell.lat.min(), cell.lat.max())
            
            if ((params.rect) and ((cnt == 0) or (params.taper_first and not params.taper_full_fg) or params.taper_both)):

            #######################################################
            # do fourier...

                if not params.dfft_first_guess:
                    freqs, uw_pmf_freqs, dat_2D_fg0 = first_guess.sappx(cell, lmbda=params.lmbda_fg, iter_solve=params.fg_iter_solve)

                    print("uw_pmf_freqs_sum:", uw_pmf_freqs.sum())

            #######################################################
            # do fourier using DFFT

                if params.dfft_first_guess:
                    ampls, uw_pmf_freqs, dat_2D_fg0, kls = first_guess.dfft(cell)
                    freqs = np.copy(ampls)

                    print("uw_pmf_freqs_sum:", uw_pmf_freqs.sum())

            #######################################################

            elif (not params.rect) and (params.cg_spsp):
                freqs, uw_pmf_freqs, dat_2D_fg0 = first_guess.sappx(cell, lmbda=params.lmbda_fg)

            elif (not params.rect):
                freqs, uw_pmf_freqs, dat_2D_fg0 = first_guess.sappx(cell, lmbda=params.lmbda_fg)

                print("uw_pmf_freqs_sum:", uw_pmf_freqs.sum())

            if params.debug_writer:
                writer.populate(idx, 'spectrum_fg', freqs)
                writer.populate(idx, 'recon_fg', dat_2D_fg0)
                writer.populate(idx, 'pmf_fg', uw_pmf_freqs)

            # plot first guess...

            if cnt == 0:
                v_extent = [dat_2D_fg0.min(), dat_2D_fg0.max()]

            if params.plot:
                fs = (15.0,4.0)
                fig, axs = plt.subplots(1,3, figsize=fs)
                fig_obj = plotter.fig_obj(fig, first_guess.fobj.nhar_i, first_guess.fobj.nhar_j)
                axs[0] = fig_obj.phys_panel(axs[0], dat_2D_fg0, title='T%i+T%i: FF reconstruction' %(idx,idx+1), xlabel='longitude [km]', ylabel='latitude [km]', extent=[cell.lon.min(), cell.lon.max(), cell.lat.min(), cell.lat.max()], v_extent=v_extent)

                if params.dfft_first_guess:
                    axs[1] = fig_obj.fft_freq_panel(axs[1], ampls, kls[0], kls[1], typ='real')
                    axs[2] = fig_obj.fft_freq_panel(axs[2], uw_pmf_freqs, kls[0], kls[1], title="PMF spectrum", typ='real')
                else:
                    axs[1] = fig_obj.freq_panel(axs[1], freqs)
                    axs[2] = fig_obj.freq_panel(axs[2], uw_pmf_freqs, title="PMF spectrum")

                plt.tight_layout()
                # plt.savefig('../output/T%i_T%i_fg.pdf' %(idx,idx+1))
                plt.show()

            ##############################################

            fq_cpy = np.copy(freqs)
            fq_cpy[np.isnan(fq_cpy)] = 0.0 # necessary. Otherwise, popping with fq_cpy.max() gives the np.nan entries first.

            if params.debug:
                total_power = fq_cpy.sum()
                print("total power =", total_power)
                print("reg max, reg min =", fq_cpy.max(), fq_cpy.min())
                print("sum(fq_cpy) =", fq_cpy.sum())

            indices = []
            max_ampls = []

            if not params.cg_spsp:
                for ii in range(params.n_modes):
                    max_idx = np.unravel_index(fq_cpy.argmax(), fq_cpy.shape)
                    indices.append(max_idx)
                    max_ampls.append(fq_cpy[max_idx])
                    max_val = fq_cpy[max_idx]
                    fq_cpy[max_idx] = 0.0
            else:
                pass

            if (params.tapering) and ((params.taper_second) or (params.taper_both)):
                utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell, topo, rect=False, padding=params.padding, topo_mask=taper.p, mask=mask_taper, filtered=False)
            else:
                utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell, topo, rect=False, filtered=False)

            # utils.get_lat_lon_segments(tri.tri_lat_verts[idx], tri.tri_lon_verts[idx], cell, topo, triangle, rect=False)

            ############################################

            if params.verbose: 
                print("top %i ampls:" %params.n_modes)
                print(max_ampls, len(max_ampls), sum(max_ampls))
                print("")
                print("top %i idxs:" %params.n_modes)
                print(indices, len(indices))

            second_guess = interface.get_pmf(nhi,nhj,params.U,params.V)

            if not params.cg_spsp:
                k_idxs = [pair[1] for pair in indices]
                l_idxs = [pair[0] for pair in indices]

                if params.dfft_first_guess:
                    second_guess.fobj.set_kls(k_idxs, l_idxs, recompute_nhij = True, components='real')
                else:
                    second_guess.fobj.set_kls(k_idxs, l_idxs, recompute_nhij = False)

                freqs, uw, dat_2D_sg0 = second_guess.sappx(cell, lmbda=params.lmbda_sg, updt_analysis=True, do_scale=True, scale=1.0, iter_solve=params.sg_iter_solve, get_n_modes=True, n_modes=params.n_modes)
            else:
                freqs = np.array(freqs, order='C')
                freqs = np.nanmean(utils.sliding_window_view(freqs, (3,3), (3,3)), axis=(-1,-2))
                freqs = np.array(freqs, order='F')

                kks = np.arange(0,nhi)[1::3]
                lls = np.arange(-nhj/2+1,nhj/2+1)[1::3]
                kklls = [kks,lls]

                freqs, uw, dat_2D_sg0 = second_guess.cg_spsp(cell, freqs, kklls, dat_2D_fg0, updt_analysis=True, scale=1.0)


            ##############################################            
            
            if params.refine:
                if params.tapering and params.taper_second:
                    utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell, topo, rect=False, padding=params.padding, filtered=False)
                    mask_taper = np.copy(cell.mask)
                utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell, topo, rect=True)
                # if ((cell.topo.shape[0] % 2) == 1):
                #     cell.topo = cell.topo[:-1,:]- dat_2D_fg0
                #     cell.lon = cell.lon[:-1]
                #     cell.lat_grid = cell.lat_grid[:-1,:]
                #     cell.lon_grid = cell.lon_grid[:-1,:]
                # elif ((cell.topo.shape[1] % 2) == 1):
                #     cell.topo = cell.topo[:,:-1]- dat_2D_fg0
                #     cell.lat = cell.lat[:-1]
                #     cell.lat_grid = cell.lat_grid[:,:-1]
                #     cell.lon_grid = cell.lon_grid[:,:-1]
                # else:
                cell.topo -= dat_2D_fg0
                cell.get_masked(mask=np.ones_like(cell.topo).astype('bool'))
                # cell.get_masked(triangle=triangle)
                cell.topo_m -= cell.topo_m.mean()
                
                first_guess = interface.get_pmf(nhi,nhj,params.U,params.V)
                if not params.dfft_first_guess:
                    freqs_fg, _, dat_2D_fg = first_guess.sappx(cell, lmbda=params.refine_lmbda_fg)
                else:
                    ampls, uw_pmf_freqs, dat_2D_fg, kls = first_guess.dfft(cell)
                    freqs_fg = np.copy(ampls)
                fq_cpy = np.copy(freqs_fg)

                indices = []
                max_ampls = []

                for ii in range(params.refine_n_modes):
                    max_idx = np.unravel_index(fq_cpy.argmax(), fq_cpy.shape)
                    indices.append(max_idx)
                    max_ampls.append(fq_cpy[max_idx])
                    max_val = fq_cpy[max_idx]
                    fq_cpy[max_idx] = 0.0
                    
                    k_idxs = [pair[1] for pair in indices]
                    l_idxs = [pair[0] for pair in indices]
                    
                if params.tapering and params.taper_second:
                    utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell, topo, rect=False, padding=params.padding, topo_mask=taper.p, mask=mask_taper, filtered=False)
                else:
                    utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell, topo, rect=False, filtered=False)
                
                second_guess = interface.get_pmf(nhi,nhj,params.U,params.V)
                if params.dfft_first_guess:
                    second_guess.fobj.set_kls(k_idxs, l_idxs, recompute_nhij = True, components='real')
                else:
                    second_guess.fobj.set_kls(k_idxs, l_idxs, recompute_nhij = False)
                
                freqs_sg, uw_sg, dat_2D_sg = second_guess.sappx(cell, lmbda=params.refine_lmbda_sg, updt_analysis= True, scale=1.0)

                # if freqs_sg.sum() < freqs.sum():
                #     if uw_sg.shape[1] > uw.shape[1]:
                #         tmp = np.zeros_like(uw_sg)
                #         tmp[:,:uw.shape[1]] = uw
                #         uw = tmp
                
                # axs_diff = freqs.shape[1] - freqs_sg.shape[1]
                # freqs_sg = np.pad(freqs_sg, ((0,0), (0,axs_diff)),mode='constant')

                freqs_tmp = freqs + freqs_sg
                cutoff = np.sort(freqs_tmp.ravel())[::-1][params.n_modes-1]
                freqs_tmp[np.where(freqs_tmp < cutoff)] = 0.0
                cell.analysis.ampls = freqs_tmp

                ideal = physics.ideal_pmf(U=params.U, V=params.V)
                uw_ref = ideal.compute_uw_pmf(cell.analysis, summed=False)

                # uw_tmp = uw + uw_ref
                # cutoff = np.sort(uw_tmp.ravel())[::-1][params.n_modes-1]
                # uw_tmp[np.where(uw_tmp < cutoff)] = 0.0

                # freqs_tmp = freqs + freqs_sg
                # freqs_tmp[np.where(uw_tmp == 0.0)] = 0.0
                # cell.analysis.ampls = freqs_tmp


                # uw += uw_sg
            
            ##############################################

            if params.plot:
                fs = (15,4.0)
                fig, axs = plt.subplots(1,3, figsize=fs)
                fig_obj = plotter.fig_obj(fig, second_guess.fobj.nhar_i, second_guess.fobj.nhar_j)
                axs[0] = fig_obj.phys_panel(axs[0], dat_2D_sg0, title='T%i: Reconstruction' %idx, xlabel='longitude [km]', ylabel='latitude [km]', extent=[cell.lon.min(), cell.lon.max(), cell.lat.min(), cell.lat.max()], v_extent=v_extent)
                if params.dfft_first_guess:
                    axs[1] = fig_obj.fft_freq_panel(axs[1], freqs, kls[0], kls[1], typ='real')
                    axs[2] = fig_obj.fft_freq_panel(axs[2], uw, kls[0], kls[1], title="PMF spectrum", typ='real')
                else:
                    axs[1] = fig_obj.freq_panel(axs[1], freqs)
                    axs[2] = fig_obj.freq_panel(axs[2], uw, title="PMF spectrum")
                plt.tight_layout()
                # plt.savefig('../output/T%i.pdf' %idx)
                plt.show()

            ##############################################

            writer.write_all(idx, cell, cell.analysis)
            writer.populate(idx, 'pmf_sg', uw)

            cell.topo = topo_orig
            cell.mask = mask_orig

            cell.uw = uw
            all_cells[cnt] = cell

            del cell
        
        cell0 = all_cells[0]
        cell1 = all_cells[1]

        if params.tapering and (params.taper_first or params.taper_both):
            cell_ref = var.topo_cell()
            utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell_ref, topo, rect=True)
        else:
            cell_ref = cell0

        ampls, uw_ref, fft_2D, kls = first_guess.dfft(cell_ref)

        ampls_sum = (all_cells[0].analysis.ampls + all_cells[1].analysis.ampls)
        all_cells[0].analysis.ampls = ampls_sum
        
        ideal = physics.ideal_pmf(U=params.U, V=params.V)
        uw_sum = ideal.compute_uw_pmf(all_cells[0].analysis)

        uw0 = all_cells[0].uw.sum()
        uw1 = all_cells[1].uw.sum()

        uw01 = 0.5 * (uw0 + uw1)

        if params.debug_writer:
            writer.populate(idx-1, 'topo_ref', cell_ref.topo)
            writer.populate(idx-1, 'spectrum_ref', ampls)
            writer.populate(idx-1, 'pmf_ref', uw_ref)

        print("")
        print("pmf tri1, tri2:", uw0, uw1)
        print("pmf ref, avg, sum:", uw_ref.sum(), uw01, uw0 + uw1)

        if params.plot:
            fs = (15,5.0)
            fig, axs = plt.subplots(1,3, figsize=fs)
            fig_obj = plotter.fig_obj(fig, second_guess.fobj.nhar_i, second_guess.fobj.nhar_j)
            axs[0] = fig_obj.phys_panel(axs[0], fft_2D, title='T%i + T%i: FFT reconstruction' %(idx-1, idx), xlabel='longitude [km]', ylabel='latitude [km]', extent=[cell0.lon.min(), cell0.lon.max(), cell0.lat.min(), cell0.lat.max()], v_extent=v_extent)

            axs[1] = fig_obj.fft_freq_panel(axs[1], ampls, kls[0], kls[1], typ='real')
            axs[2] = fig_obj.fft_freq_panel(axs[2], uw_ref, kls[0], kls[1], title="FFT PMF spectrum", typ='real')
            plt.tight_layout()
            # plt.savefig('../output/T%i_T%i_fft.pdf' %(idx-1,idx))
            plt.show()

        residual_error = (uw01 / uw_ref.sum()) - 1.0
        # residual_sum_error = ( (0.5 * uw_sum) / uw_ref.sum()) - 1.0
        residual_sum_error = ((uw0 + uw1) / uw_ref.sum()) - 1.0


        print(residual_error, residual_sum_error )


        print("")
        print("##########")
        print("")

        del all_cells

        # corrected = True
        # old_residual_error = np.copy(residual_error)
        if not corrected: 
            old_residual_error = np.copy(residual_error)


        if (tried_correction and (np.abs(residual_error) < 0.25)) or (params.no_corrections):
            corrected = True

        # MERIT x10 correction strategy
        if (residual_error > 0.0):
            params.refine = False

        # if (residual_error > 0.5):
        #     params.n_modes = int(params.n_modes / 2)
        #     params.lmbda_sg = 1e-1
        #     tried_correction = True

        # elif (residual_error > 0.2):
        #     params.n_modes = 25
        #     params.lmbda_sg = 0.05
        #     tried_correction = True

        # elif (residual_error > 0.1):
        #     params.n_modes = 50
        #     params.lmbda_sg = 1e-1
        #     tried_correction = True

        if (residual_error > 0.1):
            params.n_modes = int(params.n_modes / 2)
            # params.lmbda_sg = 1e-1
            tried_correction = True

        elif (residual_error < -0.15):
            params.refine = True
            if not hasattr(params, 'refine_n_modes'):
                params.refine_n_modes = 100
            else:
                params.refine_n_modes = int(params.refine_n_modes + 100)
            params.refine_lmbda_fg = 1e-2
            params.refine_lmbda_sg = 0.2
            tried_correction = True
        else:
            corrected = True


        if corrected:
            if np.abs(old_residual_error) < np.abs(residual_error):
                new_error = old_residual_error
            else:
                new_error = residual_error

            idx_name.append(rect_idx)
            pmf_diff.append(new_error)
            pmf_sum_diff.append(residual_sum_error)

writer.populate('decomposition', 'pmf_diff', pmf_diff)


# %%
pmf_percent_diff = np.array(pmf_sum_diff) * 100
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
