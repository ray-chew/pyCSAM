# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pycsam.src import io, var, utils, fourier
from pycsam.wrappers import interface
from pycsam.vis import plotter, cart_plot

from IPython import get_ipython

ipython = get_ipython()

if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
else:
    print(ipython)

def autoreload():
    if ipython is not None:
        ipython.run_line_magic("autoreload", "2")

from sys import exit

if __name__ != "__main__":
    exit(0)


# %%

autoreload()
from pycsam.inputs.icon_regional_run import params

if params.self_test():
    params.print()

print(params.path_compact_topo)

grid = var.grid()

# read grid
reader = io.ncdata(padding=params.padding, padding_tol=(60 - params.padding))

# writer object
writer = io.nc_writer(params)

reader.read_dat(params.path_compact_grid, grid)

clat_rad = np.copy(grid.clat)
clon_rad = np.copy(grid.clon)

grid.apply_f(utils.rad2deg)

n_cells = grid.clat.size

for c_idx in range(n_cells)[3:6]:
    # c_idx = 1
    print(c_idx)

    topo = var.topo_cell()
    lat_verts = grid.clat_vertices[c_idx]
    lon_verts = grid.clon_vertices[c_idx]

    lat_extent = [lat_verts.min() - 1.0,lat_verts.min() - 1.0,lat_verts.max() + 1.0]
    lon_extent = [lon_verts.min() - 1.0,lon_verts.min() - 1.0,lon_verts.max() + 1.0]
    # we only keep the topography that is inside this lat-lon extent.
    lat_verts = np.array(lat_extent)
    lon_verts = np.array(lon_extent)

    params.lat_extent = lat_extent
    params.lon_extent = lon_extent

    # read topography
    if not params.enable_merit:
        reader.read_dat(params.fn_topo, topo)
        reader.read_topo(topo, topo, lon_verts, lat_verts)
    else:
        reader.read_merit_topo(topo, params)
        topo.topo[np.where(topo.topo < -500.0)] = -500.0

    topo.gen_mgrids()


# %%

    clon = np.array([grid.clon[c_idx]])
    clat = np.array([grid.clat[c_idx]])
    clon_vertices = np.array([grid.clon_vertices[c_idx]])
    clat_vertices = np.array([grid.clat_vertices[c_idx]])

    ncells = 1
    nv = clon_vertices[0].size
    # -- create the triangles
    clon_vertices = np.where(clon_vertices < -180.0, clon_vertices + 360.0, clon_vertices)
    clon_vertices = np.where(clon_vertices > 180.0, clon_vertices - 360.0, clon_vertices)

    triangles = np.zeros((ncells, nv, 2))

    for i in range(0, ncells, 1):
        triangles[i, :, 0] = np.array(clon_vertices[i, :])
        triangles[i, :, 1] = np.array(clat_vertices[i, :])

    print("--> triangles done")

    if params.plot:
        cart_plot.lat_lon_icon(topo, triangles, ncells=ncells, clon=clon, clat=clat)

# %%

    print(topo.topo.shape)

# %%
    tri_idx = 0
    # initialise cell object
    cell = var.topo_cell()

    simplex_lon = triangles[tri_idx, :, 0]
    simplex_lat = triangles[tri_idx, :, 1]

    utils.get_lat_lon_segments(
        simplex_lat, simplex_lon, cell, topo, rect=params.rect
    )

    if utils.is_land(cell, simplex_lat, simplex_lon, topo):
        writer.output(c_idx, clat_rad[tri_idx], clon_rad[tri_idx], 0)
        continue
    else:
        is_land = 1

    topo_orig = np.copy(cell.topo)

    if params.dfft_first_guess:
        nhi = len(cell.lon)
        nhj = len(cell.lat)
    else:
        nhi = params.nhi
        nhj = params.nhj

    first_guess = interface.get_pmf(nhi, nhj, params.U, params.V)
    fobj_tri = fourier.f_trans(nhi, nhj)

    #######################################################
    # do fourier...

    if not params.dfft_first_guess:
        freqs, uw_pmf_freqs, dat_2D_fg0 = first_guess.sappx(cell, params.lmbda_fa)

    #######################################################
    # do fourier using DFFT

    if params.dfft_first_guess:
        ampls, uw_pmf_freqs, dat_2D_fg0, kls = first_guess.dfft(cell)
        freqs = np.copy(ampls)

    print("uw_pmf_freqs_sum:", uw_pmf_freqs.sum())

    fq_cpy = np.copy(freqs)
    fq_cpy[
        np.isnan(fq_cpy)
    ] = 0.0  # necessary. Otherwise, popping with fq_cpy.max() gives the np.nan entries first.

    indices = []
    max_ampls = []

    for ii in range(params.n_modes):
        max_idx = np.unravel_index(fq_cpy.argmax(), fq_cpy.shape)
        indices.append(max_idx)
        max_ampls.append(fq_cpy[max_idx])
        max_val = fq_cpy[max_idx]
        fq_cpy[max_idx] = 0.0

    utils.get_lat_lon_segments(
        simplex_lat, simplex_lon, cell, topo, rect=False
    )

    k_idxs = [pair[1] for pair in indices]
    l_idxs = [pair[0] for pair in indices]

    second_guess = interface.get_pmf(nhi, nhj, params.U, params.V)

    if params.dfft_first_guess:
        second_guess.fobj.set_kls(
            k_idxs, l_idxs, recompute_nhij=True, components="real"
        )
    else:
        second_guess.fobj.set_kls(k_idxs, l_idxs, recompute_nhij=False)

    freqs, uw, dat_2D_sg0 = second_guess.sappx(cell, lmbda=params.lmbda_sa, updt_analysis=True)

    cell.topo = topo_orig

    writer.output(c_idx, clat_rad[tri_idx], clon_rad[tri_idx], is_land, cell.analysis)
    
    cell.uw = uw

    if params.plot:
        fs = (15, 9.0)
        v_extent = [dat_2D_sg0.min(), dat_2D_sg0.max()]

        fig, axs = plt.subplots(2, 2, figsize=fs)

        fig_obj = plotter.fig_obj(
            fig, second_guess.fobj.nhar_i, second_guess.fobj.nhar_j
        )
        axs[0, 0] = fig_obj.phys_panel(
            axs[0, 0],
            dat_2D_sg0,
            title="T%i: Reconstruction" % tri_idx,
            xlabel="longitude [km]",
            ylabel="latitude [km]",
            extent=[cell.lon.min(), cell.lon.max(), cell.lat.min(), cell.lat.max()],
            v_extent=v_extent,
        )

        axs[0, 1] = fig_obj.phys_panel(
            axs[0, 1],
            cell.topo * cell.mask,
            title="T%i: Reconstruction" % tri_idx,
            xlabel="longitude [km]",
            ylabel="latitude [km]",
            extent=[cell.lon.min(), cell.lon.max(), cell.lat.min(), cell.lat.max()],
            v_extent=v_extent,
        )

        if params.dfft_first_guess:
            axs[1, 0] = fig_obj.fft_freq_panel(
                axs[1, 0], freqs, kls[0], kls[1], typ="real"
            )
            axs[1, 1] = fig_obj.fft_freq_panel(
                axs[1, 1], uw, kls[0], kls[1], title="PMF spectrum", typ="real"
            )
        else:
            axs[1, 0] = fig_obj.freq_panel(axs[1, 0], freqs)
            axs[1, 1] = fig_obj.freq_panel(axs[1, 1], uw, title="PMF spectrum")

        plt.tight_layout()
        plt.savefig("%sT%i.pdf" % (params.path_output, tri_idx))
        plt.show()


# %%