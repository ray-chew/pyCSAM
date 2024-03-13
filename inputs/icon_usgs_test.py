# %%
import sys
# set system path to find local modules
sys.path.append('..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import io, var, utils, fourier, physics, delaunay
from wrappers import interface
from vis import plotter, cart_plot

from sys import exit
if __name__ != "__main__": exit(0)

# %%

fn_grid = '../data/icon_compact.nc'
fn_topo = '../data/topo_compact.nc'
lat_extent = [52.,64.,64.]
lon_extent = [-141.,-158.,-127.]

tri_set = [13,104,105,106]

# Setup the Fourier parameters and object.
nhi = 24
nhj = 48

n_modes = 100

U, V = 10.0, 0.1

rect = True

debug = False
dfft_first_guess = True
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

# read topography
reader.read_dat(fn_topo, topo)

# we only keep the topography that is inside this lat-lon extent.
lat_verts = np.array(lat_extent)
lon_verts = np.array(lon_extent)

reader.read_topo(topo, topo, lon_verts, lat_verts)

topo.gen_mgrids()

# %%
#-- get coordinates and convert radians to degrees
clon = grid.clon
clat = grid.clat
clon_vertices = grid.clon_vertices
clat_vertices = grid.clat_vertices

ncells, nv = clon_vertices.shape[0], clon_vertices.shape[1]

#-- print information to stdout
print('Cells:            %6d ' % clon.size)

#-- create the triangles
clon_vertices = np.where(clon_vertices < -180., clon_vertices + 360., clon_vertices)
clon_vertices = np.where(clon_vertices >  180., clon_vertices - 360., clon_vertices)

triangles = np.zeros((ncells, nv, 2), np.float32)

for i in range(0, ncells, 1):
    triangles[i,:,0] = np.array(clon_vertices[i,:])
    triangles[i,:,1] = np.array(clat_vertices[i,:])

print('--> triangles done')


# %%
# plot ICON grid over limited-area topography
cart_plot.lat_lon_icon(topo, triangles, ncells=ncells, clon=clon, clat=clat)


# %%
idxs = []
pmfs = []

for tri_idx in tri_set:
    # initialise cell object
    cell = var.topo_cell()

    simplex_lon = triangles[tri_idx,:,0]
    simplex_lat = triangles[tri_idx,:,1]

    triangle = utils.triangle(simplex_lon, simplex_lat)
    utils.get_lat_lon_segments(simplex_lat,simplex_lon, cell, topo, triangle, rect=rect)

    topo_orig = np.copy(cell.topo)
    
    if dfft_first_guess:
        nhi = len(cell.lon)
        nhj = len(cell.lat)

    first_guess = interface.get_pmf(nhi,nhj,U,V)
    fobj_tri = fourier.f_trans(nhi,nhj)

    #######################################################
    # do fourier...

    if not dfft_first_guess:
        freqs, uw_pmf_freqs, dat_2D_fg0 = first_guess.sappx(cell, lmbda=0.0)

    #######################################################
    # do fourier using DFFT

    if dfft_first_guess:
        ampls, uw_pmf_freqs, dat_2D_fg0, kls = first_guess.dfft(cell)
        freqs = np.copy(ampls)

    print("uw_pmf_freqs_sum:", uw_pmf_freqs.sum())

    fq_cpy = np.copy(freqs)

    indices = []
    max_ampls = []

    for ii in range(n_modes):
        max_idx = np.unravel_index(fq_cpy.argmax(), fq_cpy.shape)
        indices.append(max_idx)
        max_ampls.append(fq_cpy[max_idx])
        max_val = fq_cpy[max_idx]
        fq_cpy[max_idx] = 0.0

    utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell, topo, triangle, rect=False)

    k_idxs = [pair[1] for pair in indices]
    l_idxs = [pair[0] for pair in indices]

    second_guess = interface.get_pmf(nhi,nhj,U,V)

    if dfft_first_guess:
        second_guess.fobj.set_kls(k_idxs, l_idxs, recompute_nhij = True, components='real')
    else:
        second_guess.fobj.set_kls(k_idxs, l_idxs, recompute_nhij = False)

    freqs, uw, dat_2D_sg0 = second_guess.sappx(cell, lmbda=1e-1, updt_analysis=True, scale=np.sqrt(2.0))

    cell.topo = topo_orig

    cell.uw = uw

    if plot:
        fs = (15,9.0)
        v_extent = [dat_2D_sg0.min(), dat_2D_sg0.max()]

        fig, axs = plt.subplots(2,2, figsize=fs)

        fig_obj = plotter.fig_obj(fig, second_guess.fobj.nhar_i, second_guess.fobj.nhar_j)
        axs[0,0] = fig_obj.phys_panel(axs[0,0], dat_2D_sg0, title='T%i: Reconstruction' %tri_idx, xlabel='longitude [km]', ylabel='latitude [km]', extent=[cell.lon.min(), cell.lon.max(), cell.lat.min(), cell.lat.max()], v_extent=v_extent)

        axs[0,1] = fig_obj.phys_panel(axs[0,1], cell.topo * cell.mask, title='T%i: Reconstruction' %tri_idx, xlabel='longitude [km]', ylabel='latitude [km]', extent=[cell.lon.min(), cell.lon.max(), cell.lat.min(), cell.lat.max()], v_extent=v_extent)

        if dfft_first_guess:
            axs[1,0] = fig_obj.fft_freq_panel(axs[1,0], freqs, kls[0], kls[1], typ='real')
            axs[1,1] = fig_obj.fft_freq_panel(axs[1,1], uw, kls[0], kls[1], title="PMF spectrum", typ='real')
        else:
            axs[1,0] = fig_obj.freq_panel(axs[1,0], freqs)
            axs[1,1] = fig_obj.freq_panel(axs[1,1], uw, title="PMF spectrum")

        plt.tight_layout()
        plt.savefig('../output/T%i.pdf' %tri_idx)
        plt.show()

        ideal = physics.ideal_pmf(U=U, V=V)
        uw_comp = ideal.compute_uw_pmf(cell.analysis)

        idxs.append(tri_idx)
        pmfs.append(uw_comp)
# %%
