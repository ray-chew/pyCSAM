# %%
import sys
import os
# set system path to find local modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import io, var, utils, fourier, physics, delaunay
from wrappers import interface
from vis import plotter, cart_plot


# %%
# from inputs.lam_run import params
# from inputs.selected_run import params
from inputs.iter_solve import params
from copy import deepcopy
# from inputs.debug_run import params

# print run parameters, for sanity check.
params.print()
params_orig = deepcopy(params)

# %%
# initialise data objects
grid = var.grid()
topo = var.topo_cell()

# read grid
reader = io.ncdata(padding=params.padding)

# writer object
writer = io.writer(params.output_fn, params.rect_set, debug=params.debug_writer)
writer.write_all_attrs(params)

reader.read_dat(params.fn_grid, grid)
grid.apply_f(utils.rad2deg) 

# we only keep the topography that is inside this lat-lon extent.
lat_verts = np.array(params.lat_extent)
lon_verts = np.array(params.lon_extent)

# read topography
# reader.read_dat(params.fn_topo, topo)
# reader.read_topo(topo, topo, lon_verts, lat_verts)

# path = "/scratch/atmodynamics/chew/data/MERIT/"
reader.read_merit_topo(topo, params)
topo.topo[np.where(topo.topo < -500.0)] = -500.0

topo.gen_mgrids()

tri = delaunay.get_decomposition(topo, xnp=params.delaunay_xnp, ynp=params.delaunay_ynp, padding = reader.padding)
writer.write_all('decomposition', tri)
writer.populate('decomposition', 'rect_set', params.rect_set)

# %%
# params.rect_set = delaunay.get_land_cells(tri, topo, height_tol=0.5)
# print(params.rect_set)

# %%
# Plot the loaded topography...
cart_plot.lat_lon(topo, int=1)

levels = np.linspace(-1000.0, 3000.0, 5)
cart_plot.lat_lon_delaunay(topo, tri, levels, label_idxs=True, fs=(10,6), highlight_indices=params.rect_set, output_fig=False, int=1)

# %%
del topo.lat_grid
del topo.lon_grid

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
            
                taper = utils.taper(cell, params.padding, art_it=1000)
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

            fobj_tri = fourier.f_trans(nhi,nhj)

            #######################################################

            if params.debug:
                print("cell.topo: ", cell.topo.min(), cell.topo.max())
                print("cell.lon: ", cell.lon.min(), cell.lon.max())
                print("cell.lat: ", cell.lat.min(), cell.lat.max())
            

            freqs, uw_pmf_freqs, dat_2D_fg0 = first_guess.sappx(cell, updt_analysis=True, lmbda=params.lmbda_fg, iter_solve=True, scale=np.sqrt(2.0), get_n_modes=True)

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

            ############################################## 

            uw = uw_pmf_freqs

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

        print("")
        print("pmf tri1, tri2:", uw0, uw1)
        print("pmf ref, avg, sum:", uw_ref.sum(), uw01, uw_sum)

        if params.plot:
            fs = (15,5.0)
            fig, axs = plt.subplots(1,3, figsize=fs)
            fig_obj = plotter.fig_obj(fig, first_guess.fobj.nhar_i, first_guess.fobj.nhar_j)
            axs[0] = fig_obj.phys_panel(axs[0], fft_2D, title='T%i + T%i: FFT reconstruction' %(idx-1, idx), xlabel='longitude [km]', ylabel='latitude [km]', extent=[cell0.lon.min(), cell0.lon.max(), cell0.lat.min(), cell0.lat.max()], v_extent=v_extent)

            axs[1] = fig_obj.fft_freq_panel(axs[1], ampls, kls[0], kls[1], typ='real')
            axs[2] = fig_obj.fft_freq_panel(axs[2], uw_ref, kls[0], kls[1], title="FFT PMF spectrum", typ='real')
            plt.tight_layout()
            # plt.savefig('../output/T%i_T%i_fft.pdf' %(idx-1,idx))
            plt.show()

        residual_error = (uw01 / uw_ref.sum()) - 1.0
        residual_sum_error = (uw_sum / uw_ref.sum()) - 1.0

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
            params.lmbda_sg = 1e-1
            tried_correction = True

        elif (residual_error < -0.15):
            params.refine = True
            if not hasattr(params, 'refine_n_modes'):
                params.refine_n_modes = 10
            else:
                params.refine_n_modes = int(params.refine_n_modes + 10)
            params.refine_lmbda_fg = 1e-2
            params.refine_lmbda_sg = 0.2
            tried_correction = True

        # elif (residual_error < -0.5):
        #     params.refine = True
        #     params.refine_n_modes = 100
        #     params.refine_lmbda_fg = 1e-1
        #     params.refine_lmbda_sg = 0.05
        #     tried_correction = True

        # elif (residual_error < -0.25):
        #     params.refine = True
        #     params.refine_n_modes = 80
        #     params.refine_lmbda_fg = 1e-1
        #     params.refine_lmbda_sg = 1e-1
        #     tried_correction = True

        # elif (residual_error < -0.15):
        #     params.refine = True
        #     params.refine_n_modes = 50
        #     params.refine_lmbda_fg = 1e-2
        #     params.refine_lmbda_sg = 0.2
        #     tried_correction = True
        else:
            corrected = True


        print(residual_error)

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
title = ""

print(idx_name)
print(pmf_diff)
avg_err = np.abs(pmf_diff).mean() * 100.0
print(avg_err)

pmf_percent_diff = 100.0 * np.array(pmf_diff)
data = pd.DataFrame(pmf_percent_diff,index=idx_name, columns=['values'])
fig, (ax1) = plt.subplots(1,1,sharex=True,
                         figsize=(5.0,3.0))

true_col = 'g'
false_col = 'C4' if params.dfft_first_guess else 'r'

data['values'].plot(kind='bar', width=1.0, edgecolor='black', color=(data['values'] > 0).map({True: true_col, False: false_col}))

plt.grid()

plt.xlabel("grid idx")
plt.ylabel("percentage rel. pmf diff")

err_input = np.around(avg_err,2)

if params.dfft_first_guess:
    spec_dom = "(from FFT)"
    fg_tag = 'FFT' 
else:
    spec_dom = "(%i x %i)" %(nhi,nhj)
    fg_tag = 'FF'
    
if params.refine:
    rfn_tag = ' + ext.'
else:
    rfn_tag = ''
    
cs_dd = "%s + FF%s; ~(%i x %i)km\nModes: %s; N=%i\nAverage err: " %(fg_tag, rfn_tag, params.lxkm, params.lykm, spec_dom, params.n_modes) + r"$\bf{" + str(err_input) + "\%}$"

plt.title(title, fontsize=12, pad=-10)
plt.ylim([-100,100])
plt.tight_layout()

fn = "%ix%i_%s_FF%s" %(params.lxkm, params.lykm, fg_tag, rfn_tag[:-1])
print(fn)
# plt.savefig('../output/'+fn+'_poster.pdf')
plt.show()



# %%
