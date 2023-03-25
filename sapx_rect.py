# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.ndimage.filters import gaussian_filter

from src import io, var, utils, fourier, lin_reg, reconstruction, physics, delaunay
from vis import plotter, cart_plot

import importlib

# %%
# initialise data objects
grid = var.grid()
topo = var.topo_cell()

# read grid
reader = io.ncdata()
fn = './data/icon_compact.nc'
reader.read_dat(fn, grid)
grid.apply_f(utils.rad2deg)

# read topography
fn = './data/topo_compact.nc'
reader.read_dat(fn, topo)

# we only keep the topography that is inside this lat-lon extent.
lat_verts = np.array([52.,64.,64.])
lon_verts = np.array([-141.,-158.,-127.])

reader.read_topo(topo, topo, lon_verts, lat_verts)
topo.gen_mgrids()

# Setup Delaunay triangulation domain.
tri = delaunay.get_decomposition(topo, xnp=11, ynp=11)

# levels = np.linspace(-1000.0, 3000.0, 5)
# cart_plot.lat_lon_delaunay(topo, tri, levels, label_idxs=True)

# %%
# Setup the Fourier parameters and object.
nhi = 24
nhj = 24

U, V = 10.0, 0.1
AE = 6373.0 * 1E3

debug = False
debug_first_guess = False
plot = False

#11x6
# rect_set = np.sort([0,4,54,92,52,16,44,48,88,58,94])

#11x11
rect_set = np.sort([20,26,4,74,130,168,180,102,40,112])
rect_set = np.array([102])

# rect_set = np.sort([0,4])
# rect_set = np.sort([320])
# xnp = 22, ynp = 22

#22x22
# rect_set = np.sort([88,320,126,392,714,262,732,784,112])
# rect_set = np.array([392])
print("rect_set = ", rect_set)
print("")

pmf_diff = []
pmf_sum_diff = []
idx_name = []

for rect_idx in rect_set:
    all_cells = np.zeros(2, dtype='object')
    iter_cnt = 0
    n_modes = 100
    n_change = -np.inf
    errs = []
    success = False
    while not success:
        for cnt, idx in enumerate(range(rect_idx,rect_idx+2)):
            # initialise cell object
            cell = var.topo_cell()

            print("computing idx:", idx)

            simplex_lat = tri.tri_lat_verts[idx]
            simplex_lon = tri.tri_lon_verts[idx]

            triangle = utils.triangle(simplex_lon, simplex_lat)
            utils.get_lat_lon_segments(tri.tri_lat_verts[idx], tri.tri_lon_verts[idx], cell, topo, triangle, rect=True)

            nhi = len(cell.lon)
            nhj = len(cell.lat)
            fobj = fourier.f_trans(nhi,nhj)
            fobj_tri = fourier.f_trans(nhi,nhj)

 
            #######################################################
            # do we run idealised? 
            
#             if ((cnt == 0)):
#                 cell.topo[...] = 0.0
#                 cell.topo = sinusoidal_basis(cell)
#                 tmp = np.copy(cell.topo)
#                 # cell.mask[...] = True
#                 cell.get_masked(triangle, mask=cell.mask)

#             elif ((cnt == 1)):
#                 cell.topo = tmp
#                 # cell.mask[...] = True
#                 cell.get_masked(triangle, mask=cell.mask)

            #######################################################
    
            if debug:
                print("cell.topo: ", cell.topo.min(), cell.topo.max())
                print("cell.lon: ", cell.lon.min(), cell.lon.max())
                print("cell.lat: ", cell.lat.min(), cell.lat.max())

            mask_tmp = np.copy(cell.mask)

            #######################################################
            # do fourier...

#             fobj.do_full(cell)
#             am, data_recons = lin_reg.do(fobj, cell, lmbda = 0.0)

#             if debug: print("data_recons: ", data_recons.min(), data_recons.max())

#             dat_2D = reconstruction.recon_2D(data_recons, cell)

#             if debug: print("dat_2D: ", dat_2D.min(), dat_2D.max())

#             fobj.get_freq_grid(am)
#             freqs = np.abs(fobj.ampls)
            
#             print(np.sort(ampls.reshape(-1,))[::-1][:25])
            
#             # print(np.sort(freqs.reshape(-1,))[::-1][:25])

#             analysis = var.analysis()
#             analysis.get_attrs(fobj, freqs)

#             ideal = physics.ideal_pmf(U=U, V=V)

#             uw_pmf_freqs = ideal.compute_uw_pmf(analysis, summed=False)

#             print("uw_pmf_freqs_sum:", uw_pmf_freqs.sum())

            #######################################################
            # do fourier using DFFT

            ampls = np.fft.rfft2(cell.topo)
            ampls /= ampls.size

            freqs = ampls

            wlat = np.diff(cell.lat).max()
            wlon = np.diff(cell.lon).max()

            sz = cell.topo.size

            kks = np.fft.rfftfreq((ampls.shape[1] * 2) - 1, d=1.0)#.reshape(-1,1)
            lls = np.fft.fftfreq((ampls.shape[0]), d=1.0)#.reshape(1,-1)
            
            ampls = np.fft.fftshift(ampls)
            # kks = np.fft.fftshift(kks)
            lls = np.fft.fftshift(lls)
            
            # print(kks, lls)

            kkg, llg = np.meshgrid(kks, lls)
            
            dat_2D = np.fft.ifft2(np.fft.ifftshift(ampls) * ampls.size).real 
            ampls = np.abs(ampls)
            
            print(np.sort(ampls.reshape(-1,))[::-1][:25])

            analysis = var.analysis()
            analysis.wlat = wlat
            analysis.wlon = wlon
            analysis.ampls = ampls
            analysis.kks = kkg#.reshape(-1,)#[1:] #/ kkg.size
            analysis.lls = llg#.reshape(-1,)#[1:] #/ llg.size

            ideal = physics.ideal_pmf(U=U, V=V)
            uw_pmf_freqs = ideal.compute_uw_pmf(analysis, summed=False)

            print("uw_pmf_freqs_sum:", uw_pmf_freqs.sum())

            #######################################################

            if cnt == 0:
                v_extent = [dat_2D.min(), dat_2D.max()]

            if plot:
                fs = (15.0,5.0)
                fig, axs = plt.subplots(1,3, figsize=fs)
                fig_obj = plotter.fig_obj(fig, fobj.nhar_i, fobj.nhar_j)
                axs[0] = fig_obj.phys_panel(axs[0], dat_2D, title='T%i: Reconstruction' %idx, xlabel='longitude', ylabel='latitude', extent=[cell.lon.min(), cell.lon.max(), cell.lat.min(), cell.lat.max()], v_extent=v_extent)
                # axs[1] = fig_obj.freq_panel(axs[1], freqs)
                # axs[2] = fig_obj.freq_panel(axs[2], uw_pmf_freqs, title="PMF spectrum")
                axs[1] = fig_obj.fft_freq_panel(axs[1], freqs, kks, lls)
                axs[2] = fig_obj.fft_freq_panel(axs[2], uw_pmf_freqs, kks, lls, title="PMF spectrum")
                plt.tight_layout()
                # plt.savefig('output/T%i.pdf' %idx)
                plt.show()


            ##############################################
            # debug: compute amplitudes of the first guess
            if debug_first_guess:
                ampls = np.fft.rfft2(cell.topo) 
                ampls /= ampls.size
                ampls = np.abs(ampls)

                ref_power = ampls.sum()

                print("ampls first guess:")
                print(np.sort(ampls.reshape(-1,))[::-1][:25])

                wlat = np.diff(cell.lat).max()
                wlon = np.diff(cell.lon).max()

                sz = cell.topo.size

                # kks = np.fft.fftfreq(cell.topo.shape[1])
                # lls = np.fft.fftfreq(cell.topo.shape[0])

                kks = np.fft.rfftfreq((ampls.shape[1] * 2) - 1, d=1.0).reshape(-1,1)
                lls = np.fft.fftfreq((ampls.shape[0]), d=1.0).reshape(1,-1)

                kkg, llg = np.meshgrid(kks, lls)

                analysis = var.analysis()
                analysis.wlat = wlat
                analysis.wlon = wlon
                analysis.ampls = ampls
                analysis.kks = kkg#.reshape(-1,)#[1:] #/ kkg.size
                analysis.lls = llg#.reshape(-1,)#[1:] #/ llg.size

                print(ampls.shape, kks.shape, lls.shape)

                ideal = physics.ideal_pmf(U=U,V=V)
                uw_ref = ideal.compute_uw_pmf(analysis, summed=False)

                print("uw_ref:", uw_ref.sum())
            ##############################################

            fq_cpy = np.copy(freqs)
            # fq_cpy = uw_pmf_freqs
            # total_power = fq_cpy.sum()
            total_power = freqs.sum()

            # ref_power = np.abs(np.fft.rfft2(cell.topo - cell.topo.mean())) / cell.topo.size
            # ref_power = ref_power.sum()
            # print("ref power =", ref_power)

            if debug:
                print("ref power =", ref_power)
                print("total power =", total_power)
                print("reg max, reg min =", fq_cpy.max(), fq_cpy.min())
                print("sum(fq_cpy) =", fq_cpy.sum())
            # print("ref power =", ref_power)

            indices = []
            max_ampls = []

            for ii in range(n_modes):
                max_idx = np.unravel_index(fq_cpy.argmax(), fq_cpy.shape)
                indices.append(max_idx)
                max_ampls.append(fq_cpy[max_idx])
                max_val = fq_cpy[max_idx]
                fq_cpy[max_idx] = 0.0

                # if (sum(max_ampls) >= 0.1 * ref_power):
                    # break
                # if (sum(max_ampls) >= 1.0 * total_power):
                    # break
                # if sum(max_ampls) >= (total_power / 44000) * 44000:
                    # break
                # if max_val < 0.01 * total_power:
                # if max_ampls[-1] <= 0.1 * max_ampls[0]:
                    # break

            utils.get_lat_lon_segments(tri.tri_lat_verts[idx], tri.tri_lon_verts[idx], cell, topo, triangle, rect=False)

            ############################################
            # cell.topo = tmp
            # triangle = utils.triangle(tri.tri_lon_verts[idx], tri.tri_lat_verts[idx])
            # cell.get_masked(triangle)
            ############################################

            # if debug: 
            print("top %i ampls:" %n_modes)
            print(max_ampls, len(max_ampls), sum(max_ampls))
            print("")
            print("top %i idxs:" %n_modes)
            print(indices, len(indices))

            k_idxs = [pair[1] for pair in indices]
            l_idxs = [pair[0] for pair in indices]
            
            fobj_tri.set_kls(k_idxs, l_idxs, recompute_nhij = True)
            fobj_tri.do_full(cell)

            am, data_recons = lin_reg.do(fobj_tri, cell, lmbda = 1e-1)

            fobj_tri.get_freq_grid(am)
            dat_2D = reconstruction.recon_2D(data_recons, cell)

            freqs = np.abs(fobj_tri.ampls)

            if debug: print("\n double reg. sum: ",freqs.sum())

            analysis = var.analysis()
            analysis.get_attrs(fobj_tri, freqs)
            analysis.recon = dat_2D
            analysis.max_ampls = max_ampls

            cell.analysis = analysis

            uw = ideal.compute_uw_pmf(cell.analysis, summed=False)
            cell.uw = uw
            
            # topo_tri = cell.topo * cell.mask
            # topo_tri -= topo_tri.mean()

            all_cells[cnt] = cell

            if plot:
                fs = (15,5)
                fig, axs = plt.subplots(1,3, figsize=fs)
                fig_obj = plotter.fig_obj(fig, fobj_tri.nhar_i, fobj_tri.nhar_j)
                axs[0] = fig_obj.phys_panel(axs[0], dat_2D, title='T%i: Reconstruction' %idx, xlabel='longitude', ylabel='latitude', extent=[cell.lon.min(), cell.lon.max(), cell.lat.min(), cell.lat.max()], v_extent=v_extent)
                # axs[1] = fig_obj.freq_panel(axs[1], freqs)
                # axs[2] = fig_obj.freq_panel(axs[2], uw, title="PMF spectrum")
                axs[1] = fig_obj.fft_freq_panel(axs[1], freqs, kks, lls)
                axs[2] = fig_obj.fft_freq_panel(axs[2], uw, kks, lls, title="PMF spectrum")
                plt.tight_layout()
                plt.savefig('output/T%i.pdf' %idx)
                plt.show()

        cell0 = all_cells[0]
        cell1 = all_cells[1]

        topo_blur = gaussian_filter(cell0.topo - cell0.topo.mean(), sigma=0)

        fft_freqs = np.fft.fft2(topo_blur)
        ampls = np.copy(fft_freqs / fft_freqs.size)
        # ampls = np.abs(fft_freqs / fft_freqs.size)

        # for now, we artifically initialise analysis object.
        # wlat = np.diff(cell.lat).max()
        # wlon = np.diff(cell.lon).max()
        wlat = cell.wlat
        wlon = cell.wlon

        kks = np.fft.fftfreq(cell.topo.shape[1])
        lls = np.fft.fftfreq(cell.topo.shape[0])
        
        ampls = np.fft.fftshift(ampls)
        kks = np.fft.fftshift(kks)
        lls = np.fft.fftshift(lls)

        # kks = np.fft.rfftfreq((ampls.shape[1] * 2) - 1, d=1.0).reshape(-1,1)
        # lls = np.fft.fftfreq((ampls.shape[0]), d=1.0).reshape(1,-1)

        kkg, llg = np.meshgrid(kks, lls)

        # kls = ((2.0 * np.pi * kkg/wlon)**2 + (2.0 * np.pi * llg/wlat)**2)**0.5        
        # print(kls.reshape(-1,)[:25])
        # print((((2.0 * np.pi / 5000))**0.5))
        # print(np.exp(-(kls / (2.0 * np.pi / 5000))**2.0))

        # ampls *= np.exp(-(kls / (2.0 * np.pi / 5000))**2.0)
        
        fft_2D = np.fft.ifft2(np.fft.ifftshift(ampls) * ampls.size).real #- topo_blur
        ampls = np.abs(ampls)
        
        print("ref_power:", ampls.sum())
        print(np.sort(ampls.reshape(-1,))[::-1][:25])

        analysis = var.analysis()
        analysis.wlat = wlat
        analysis.wlon = wlon
        analysis.ampls = ampls
        analysis.kks = kkg#.reshape(-1,)#[1:] #/ kkg.size
        analysis.lls = llg#.reshape(-1,)#[1:] #/ llg.size

        ideal = physics.ideal_pmf(U=U, V=V)
        uw_ref = ideal.compute_uw_pmf(analysis, summed=False)

        # uw0 = ideal.compute_uw_pmf(all_cells[0].analysis) #* all_cells[0].topo_m.size
        # uw1 = ideal.compute_uw_pmf(all_cells[1].analysis) #* all_cells[1].topo_m.size
        ampls_sum = (all_cells[0].analysis.ampls + all_cells[1].analysis.ampls)
        all_cells[0].analysis.ampls = ampls_sum
        uw_sum = ideal.compute_uw_pmf(all_cells[0].analysis)
        
        uw0 = all_cells[0].uw.sum()
        uw1 = all_cells[1].uw.sum()

        uw01 = 0.5 * (uw0 + uw1)
        print("pmf tri1, tri2:", uw0, uw1)
        print("pmf ref, avg, sum:", uw_ref.sum(), uw01, uw_sum)
        
        if plot:
            fs = (15,5)
            fig, axs = plt.subplots(1,3, figsize=fs)
            fig_obj = plotter.fig_obj(fig, fobj_tri.nhar_i, fobj_tri.nhar_j)
            axs[0] = fig_obj.phys_panel(axs[0], fft_2D, title='T%i + T%i: FFT reconstruction' %(idx-1, idx), xlabel='longitude', ylabel='latitude', extent=[cell.lon.min(), cell.lon.max(), cell.lat.min(), cell.lat.max()], v_extent=v_extent)
            axs[1] = fig_obj.fft_freq_panel(axs[1], ampls, kks, lls)
            axs[2] = fig_obj.fft_freq_panel(axs[2], uw_ref, kks, lls, title="FFT PMF spectrum")
            plt.tight_layout()
            plt.savefig('output/T%i.pdf' %idx)
            plt.show()

        # residual_error = (uw01 - uw_ref.sum()) / uw_ref.sum() 
        success = True
        residual_error = (uw01 / uw_ref.sum()) - 1.0
        residual_sum_error = (uw_sum / uw_ref.sum()) - 1.0
        n_change_last = np.copy(n_change)
        
        n_change = int(np.abs(residual_error) * 10.0)
        if n_change <= 3:
            n_change = 1
        elif n_change == n_change_last:
            n_change -= 1
        elif n_change > 3 and iter_cnt > 10:
            n_change = 1
            
        if n_change >= n_modes:
            n_change = int(n_modes/2)
        
        print(iter_cnt + 1, (residual_error * 100.0), n_modes)

        errs.append(residual_error)
        if iter_cnt > 2:
            if (errs[-1] == errs[-3]) or iter_cnt > 30:
                residual_error = np.array(np.abs(errs)).min()
                print("uw_pmf (cell 0):", cell0.analysis.max_ampls)
                print("uw_pmf (cell 1):", cell1.analysis.max_ampls)
                print("")
                break
                
        if np.abs(residual_error * 100.0) < 10.0:
            success = True
            print("uw_pmf (cell 0):", cell0.analysis.max_ampls)
            print("uw_pmf (cell 1):", cell1.analysis.max_ampls)
            print("")
            
        elif (residual_error * 100.0) > 0.0:
            n_modes -= max(n_change, 1)
            iter_cnt += 1
        elif (residual_error * 100.0) < 0.0:
            n_modes += max(n_change, 1)
            iter_cnt += 1
            
        print("")
            
    idx_name.append(rect_idx)
    pmf_diff.append(residual_error)
    pmf_sum_diff.append(residual_sum_error)
# %%
