# %%
import sys
# set system path to find local modules
sys.path.append('..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import io, var, utils, fourier, physics, delaunay
from runs import interface
from vis import plotter, cart_plot


# %%
# fn_grid = '/scratch/atmodynamics/chew/data/icon_compact.nc'
# fn_topo = '/scratch/atmodynamics/chew/data/topo_compact.nc'

fn_grid = '/home/ray/git-projects/spec_appx/data/icon_compact.nc'
fn_topo = '/home/ray/git-projects/spec_appx/data/topo_compact.nc'

lat_extent = [52.,64.,64.]
lon_extent = [-141.,-158.,-127.]

delaunay_xnp = 16
delaunay_ynp = 11
rect_set = np.sort([156,154,32,72,68,160,96,162,276,60])
rect_set = np.sort([156,154,32,72,160,96,162,276])
# rect_set = np.sort([52,62,110,280,296,298,178,276,244,242])
# rect_set = np.sort([276])
lxkm, lykm = 120, 120

# Setup the Fourier parameters and object.
nhi = 24
nhj = 48

n_modes = 100

U, V = 10.0, 0.1

cg_spsp = False # coarse grain the spectral space?
rect = False if cg_spsp else True 

tapering = True
rect = True
padding=10


debug = False
dfft_first_guess = False
refine = False
verbose = False

plot = True


# %%
# initialise data objects
grid = var.grid()
topo = var.topo_cell()

# read grid
reader = io.ncdata()

reader.read_dat(fn_grid, grid)
grid.apply_f(utils.rad2deg) 

# we only keep the topography that is inside this lat-lon extent.
lat_verts = np.array(lat_extent)
lon_verts = np.array(lon_extent)

# read topography
reader.read_dat(fn_topo, topo)
reader.read_topo(topo, topo, lon_verts, lat_verts)

# path = "/scratch/atmodynamics/chew/data/MERIT/"
# reader.read_merit_topo(topo, path, lat_verts, lon_verts)
# topo.topo[np.where(topo.topo < -100)] = -100

topo.gen_mgrids()

tri = delaunay.get_decomposition(topo, xnp=delaunay_xnp, ynp=delaunay_ynp)

pmf_diff = []
pmf_sum_diff = []
idx_name = []

# %%
# Plot the loaded topography...
cart_plot.lat_lon(topo, int=20)

levels = np.linspace(-1000.0, 3000.0, 5)
cart_plot.lat_lon_delaunay(topo, tri, levels, label_idxs=True, fs=(10,6), highlight_indices=rect_set, output_fig=False, int=20)

# %%
del topo.lat_grid
del topo.lon_grid

# %%
for rect_idx in rect_set:
    all_cells = np.zeros(2, dtype='object')
    for cnt, idx in enumerate(range(rect_idx,rect_idx+2)):
        # initialise cell object
        cell = var.topo_cell()

        print("computing idx:", idx)

        simplex_lat = tri.tri_lat_verts[idx]
        simplex_lon = tri.tri_lon_verts[idx]

        if tapering:
            utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell, topo, rect=False)
        
            taper = utils.taper(cell, padding, art_it=1000)
            taper.do_tapering()

            utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell, topo, rect=False, padding=padding)

            mask_taper = np.copy(cell.mask)

        # else:
        utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell, topo, rect=rect)

        topo_orig = np.copy(cell.topo)
        mask_orig = np.copy(cell.mask)
        
        if dfft_first_guess:
            nhi = len(cell.lon)
            nhj = len(cell.lat)

        first_guess = interface.get_pmf(nhi,nhj,U,V)

        fobj_tri = fourier.f_trans(nhi,nhj)

        #######################################################

        if debug:
            print("cell.topo: ", cell.topo.min(), cell.topo.max())
            print("cell.lon: ", cell.lon.min(), cell.lon.max())
            print("cell.lat: ", cell.lat.min(), cell.lat.max())
        
        if ((cnt == 0) and (rect)):

        #######################################################
        # do fourier...

            if not dfft_first_guess:
                freqs, uw_pmf_freqs, dat_2D_fg0 = first_guess.sappx(cell, lmbda=1e-2)

                print("uw_pmf_freqs_sum:", uw_pmf_freqs.sum())

        #######################################################
        # do fourier using DFFT

            if dfft_first_guess:
                ampls, uw_pmf_freqs, dat_2D_fg0, kls = first_guess.dfft(cell)
                freqs = np.copy(ampls)

                print("uw_pmf_freqs_sum:", uw_pmf_freqs.sum())

        #######################################################

        elif (not rect) and (cg_spsp):
            freqs, uw_pmf_freqs, dat_2D_fg0 = first_guess.sappx(cell, lmbda=1e-1)

        elif (not rect):
            freqs, uw_pmf_freqs, dat_2D_fg0 = first_guess.sappx(cell, lmbda=1e-2)

            print("uw_pmf_freqs_sum:", uw_pmf_freqs.sum())

        # plot first guess...

        if cnt == 0:
            v_extent = [dat_2D_fg0.min(), dat_2D_fg0.max()]

        if plot:
            fs = (15.0,4.0)
            fig, axs = plt.subplots(1,3, figsize=fs)
            fig_obj = plotter.fig_obj(fig, first_guess.fobj.nhar_i, first_guess.fobj.nhar_j)
            axs[0] = fig_obj.phys_panel(axs[0], dat_2D_fg0, title='T%i+T%i: FF reconstruction' %(idx,idx+1), xlabel='longitude [km]', ylabel='latitude [km]', extent=[cell.lon.min(), cell.lon.max(), cell.lat.min(), cell.lat.max()], v_extent=v_extent)

            if dfft_first_guess:
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

        if debug:
            total_power = fq_cpy.sum()
            print("total power =", total_power)
            print("reg max, reg min =", fq_cpy.max(), fq_cpy.min())
            print("sum(fq_cpy) =", fq_cpy.sum())

        indices = []
        max_ampls = []

        if not cg_spsp:
            for ii in range(n_modes):
                max_idx = np.unravel_index(fq_cpy.argmax(), fq_cpy.shape)
                indices.append(max_idx)
                max_ampls.append(fq_cpy[max_idx])
                max_val = fq_cpy[max_idx]
                fq_cpy[max_idx] = 0.0
        else:
            pass

        if tapering:
            utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell, topo, rect=False, padding=padding, topo_mask=taper.p, mask=mask_taper)
        else:
            utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell, topo, rect=False)

        # utils.get_lat_lon_segments(tri.tri_lat_verts[idx], tri.tri_lon_verts[idx], cell, topo, triangle, rect=False)

        ############################################

        if verbose: 
            print("top %i ampls:" %n_modes)
            print(max_ampls, len(max_ampls), sum(max_ampls))
            print("")
            print("top %i idxs:" %n_modes)
            print(indices, len(indices))

        second_guess = interface.get_pmf(nhi,nhj,U,V)

        if not cg_spsp:
            k_idxs = [pair[1] for pair in indices]
            l_idxs = [pair[0] for pair in indices]

            if dfft_first_guess:
                second_guess.fobj.set_kls(k_idxs, l_idxs, recompute_nhij = True, components='real')
            else:
                second_guess.fobj.set_kls(k_idxs, l_idxs, recompute_nhij = False)

            freqs, uw, dat_2D_sg0 = second_guess.sappx(cell, lmbda=1e-1, updt_analysis=True, scale=np.sqrt(2.0))
        else:
            freqs = np.array(freqs, order='C')
            freqs = np.nanmean(utils.sliding_window_view(freqs, (3,3), (3,3)), axis=(-1,-2))
            freqs = np.array(freqs, order='F')

            kks = np.arange(0,nhi)[1::3]
            lls = np.arange(-nhj/2+1,nhj/2+1)[1::3]
            kklls = [kks,lls]

            freqs, uw, dat_2D_sg0 = second_guess.cg_spsp(cell, freqs, kklls, dat_2D_fg0, updt_analysis=True, scale=np.sqrt(2.0))


        ##############################################            
        
        if refine:
            utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell, topo, rect=True, filtered=False)
            cell.topo -= dat_2D_fg0
            cell.get_masked(mask=np.ones_like(cell.topo).astype('bool'))
            # cell.get_masked(triangle=triangle)
            cell.topo_m -= cell.topo_m.mean()
            
            first_guess = interface.get_pmf(nhi,nhj,U,V)
            if not dfft_first_guess:
                freqs_fg, _, dat_2D_fg = first_guess.sappx(cell, lmbda=0.0)
            else:
                ampls, uw_pmf_freqs, dat_2D_fg, kls = first_guess.dfft(cell)
                freqs_fg = np.copy(ampls)
            fq_cpy = np.copy(freqs_fg)

            indices = []
            max_ampls = []

            for ii in range(25):
                max_idx = np.unravel_index(fq_cpy.argmax(), fq_cpy.shape)
                indices.append(max_idx)
                max_ampls.append(fq_cpy[max_idx])
                max_val = fq_cpy[max_idx]
                fq_cpy[max_idx] = 0.0
                
                k_idxs = [pair[1] for pair in indices]
                l_idxs = [pair[0] for pair in indices]
                
            utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell, topo, rect=False)
            
            second_guess = interface.get_pmf(nhi,nhj,U,V)
            if dfft_first_guess:
                second_guess.fobj.set_kls(k_idxs, l_idxs, recompute_nhij = True, components='real')
            else:
                second_guess.fobj.set_kls(k_idxs, l_idxs, recompute_nhij = False)
            
            freqs_sg, uw_sg, dat_2D_sg = second_guess.sappx(cell, lmbda=0.2, updt_analysis= True, scale=np.sqrt(2.0))

            if freqs_sg.sum() < freqs.sum():
                if uw_sg.shape[1] > uw.shape[1]:
                    tmp = np.zeros_like(uw_sg)
                    tmp[:,:uw.shape[1]] = uw
                    uw = tmp
                uw += uw_sg
        
        ##############################################

        if plot:
            fs = (15,4.0)
            fig, axs = plt.subplots(1,3, figsize=fs)
            fig_obj = plotter.fig_obj(fig, second_guess.fobj.nhar_i, second_guess.fobj.nhar_j)
            axs[0] = fig_obj.phys_panel(axs[0], dat_2D_sg0, title='T%i: Reconstruction' %idx, xlabel='longitude [km]', ylabel='latitude [km]', extent=[cell.lon.min(), cell.lon.max(), cell.lat.min(), cell.lat.max()], v_extent=v_extent)
            if dfft_first_guess:
                axs[1] = fig_obj.fft_freq_panel(axs[1], freqs, kls[0], kls[1], typ='real')
                axs[2] = fig_obj.fft_freq_panel(axs[2], uw, kls[0], kls[1], title="PMF spectrum", typ='real')
            else:
                axs[1] = fig_obj.freq_panel(axs[1], freqs)
                axs[2] = fig_obj.freq_panel(axs[2], uw, title="PMF spectrum")
            plt.tight_layout()
            # plt.savefig('../output/T%i.pdf' %idx)
            plt.show()

        ##############################################

        cell.topo = topo_orig
        cell.mask = mask_orig

        cell.uw = uw
        all_cells[cnt] = cell

        del cell
    
    cell0 = all_cells[0]
    cell1 = all_cells[1]

    if not rect:
        cell_ref = var.topo_cell()
        utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell_ref, topo, rect=True)
    else:
        cell_ref = cell0

    ampls, uw_ref, fft_2D, kls = first_guess.dfft(cell_ref)

    ampls_sum = (all_cells[0].analysis.ampls + all_cells[1].analysis.ampls)
    all_cells[0].analysis.ampls = ampls_sum
    
    ideal = physics.ideal_pmf(U=U, V=V)
    uw_sum = ideal.compute_uw_pmf(all_cells[0].analysis)

    uw0 = all_cells[0].uw.sum()
    uw1 = all_cells[1].uw.sum()

    uw01 = 0.5 * (uw0 + uw1)

    print("")
    print("pmf tri1, tri2:", uw0, uw1)
    print("pmf ref, avg, sum:", uw_ref.sum(), uw01, uw_sum)

    if plot:
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
    residual_sum_error = (uw_sum / uw_ref.sum()) - 1.0

    print("")
    print("##########")
    print("")
            
    idx_name.append(rect_idx)
    pmf_diff.append(residual_error)
    pmf_sum_diff.append(residual_sum_error)

    del all_cells

# %%
title = ""

print(idx_name)
print(pmf_diff)
avg_err = np.abs(pmf_diff).mean() * 100.0
print(avg_err)

pmf_percent_diff = 100.0 * np.array(pmf_diff)
data = pd.DataFrame(pmf_percent_diff,index=idx_name, columns=['values'])
fig, (ax1) = plt.subplots(1,1,sharex=True,
                         figsize=(3.0,2.0))

true_col = 'g'
false_col = 'C4' if dfft_first_guess else 'r'

data['values'].plot(kind='bar', width=1.0, edgecolor='black', color=(data['values'] > 0).map({True: true_col, False: false_col}))

plt.xlabel("grid idx")
plt.ylabel("percentage rel. pmf diff")

err_input = np.around(avg_err,2)

if dfft_first_guess:
    spec_dom = "(from FFT)"
    fg_tag = 'FFT' 
else:
    spec_dom = "(%i x %i)" %(nhi,nhj)
    fg_tag = 'FF'
    
if refine:
    rfn_tag = ' + ext.'
else:
    rfn_tag = ''
    
cs_dd = "%s + FF%s; ~(%i x %i)km\nModes: %s; N=%i\nAverage err: " %(fg_tag, rfn_tag, lxkm, lykm, spec_dom, n_modes) + r"$\bf{" + str(err_input) + "\%}$"

plt.title(title, fontsize=12, pad=-10)
plt.ylim([-50,50])
plt.tight_layout()

fn = "%ix%i_%s_FF%s" %(lxkm, lykm, fg_tag, rfn_tag[:-1])
print(fn)
# plt.savefig('../output/'+fn+'_poster.pdf')
plt.show()



# %%
