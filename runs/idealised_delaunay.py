# %%
import noise
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy

import sys
import os
# set system path to find local modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src import utils, var
from wrappers import interface, diagnostics

from IPython import get_ipython
ipython = get_ipython()

if '__IPYTHON__' in globals():
    ipython.run_line_magic('load_ext autoreload')

def autoreload():
    if '__IPYTHON__' in globals():
        ipython.run_line_magic('autoreload')

autoreload()


# %%
# ref: https://jackmckew.dev/3d-terrain-in-python.html
res_x = res_y = 240
scale_fac = 2000.0

shape = (res_x,res_y)
scale = 60.0
octaves = 6
persistence = 0.5
lacunarity = 2.0

world = np.zeros(shape)
for i in range(shape[0]):
    for j in range(shape[1]):
        world[i][j] = noise.pnoise2(i/scale, 
                                    j/scale, 
                                    octaves=octaves, 
                                    persistence=persistence, 
                                    lacunarity=lacunarity, 
                                    repeatx=1024, 
                                    repeaty=1024, 
                                    base=42)

world -= world.mean()
world /= world.max()
world *= scale_fac

xx = np.linspace(0, 2.0 * np.pi * scale_fac, res_x)
X, Y = np.meshgrid(xx,xx)
kl = 1.0 / scale_fac #2.0 * np.pi

bg_terrain = np.zeros(shape)
bg = -(scale_fac / 2.0) * (np.cos(kl * X + kl * Y))

total = world# + bg
# total = bg

plt.imshow(world,cmap='terrain', origin='lower')
plt.colorbar()
plt.show()

plt.imshow(bg, origin='lower')
plt.colorbar()
plt.show()

plt.imshow(total - total.mean(), cmap='terrain', origin='lower')
plt.colorbar()
plt.show()


# %%
U, V = 10.0, 0.0
nhi, nhj = 24, 48

run = interface.get_pmf(nhi, nhj, U, V)

grid = var.grid()
cell = var.topo_cell()
cell.topo = total

# vid = utils.delaunay(grid, cell, ymax = 2.0*np.pi*scale_fac, xmax = 2.0*np.pi*scale_fac, res_x=res_x, res_y=res_y)

vid = utils.isosceles(grid, cell, ymax = 2.0*np.pi*scale_fac, xmax = 2.0*np.pi*scale_fac, res=res_x)

lat_v = grid.clat_vertices[vid,:]
lon_v = grid.clon_vertices[vid,:]

cell.gen_mgrids()

triangle = utils.gen_triangle(lon_v, lat_v)
cell.get_masked(mask=np.ones_like(cell.topo).astype('bool'))
# cell.get_masked(triangle=triangle)
cell.wlat = np.diff(cell.lat).mean()
cell.wlon = np.diff(cell.lon).mean()

ampls_ref, uw_ref, fft_2D_ref, kls_ref = run.dfft(cell)
sols = (cell, ampls_ref, uw_ref, fft_2D_ref)
v_extent = [fft_2D_ref.min(), fft_2D_ref.max()]

print(ampls_ref.shape)
max_idx = np.unravel_index(ampls_ref.argmax(), ampls_ref.shape)
print(max_idx)
print(ampls_ref.max())


params = var.params()
params.plot = True
params.lmbda_fa = 0.0
params.lmbda_sa = 0.01
dplot = diagnostics.diag_plotter(params, nhi, nhj)
dplot.show((0,1), sols, kls=kls_ref, v_extent = v_extent, dfft_plot=True)

print(uw_ref.sum())




# %%
autoreload()

delaunay_decomposition = True
do_rhs_recomputation = True

topo = var.topo_cell()
topo.topo = total
topo.lat = cell.lat
topo.lon = cell.lon

params.dfft_first_guess = False
params.n_modes = 100

###########################################
# cell_fa
###########################################

cell = var.topo_cell()
cell.topo = total


if delaunay_decomposition:
    vid = utils.delaunay(grid, cell, ymax = 2.0*np.pi*scale_fac, xmax = 2.0*np.pi*scale_fac, res_x=res_x, res_y=res_y)
else:
    vid = utils.isosceles(grid, cell, ymax = 2.0*np.pi*scale_fac, xmax = 2.0*np.pi*scale_fac, res=res_x)

lat_v = grid.clat_vertices[vid,:]
lon_v = grid.clon_vertices[vid,:]

cell.gen_mgrids()
cell.get_masked(mask=np.ones_like(cell.topo).astype('bool'))
cell.wlat = np.diff(cell.lat).mean()
cell.wlon = np.diff(cell.lon).mean()

if params.dfft_first_guess:
    nhi = len(cell.lon)
    nhj = len(cell.lat)
    ampls_fa = np.copy(ampls_ref)
else:
    first_guess = interface.get_pmf(nhi, nhj, U, V)

    ampls_fa, uw_fa, dat_2D_fa = first_guess.sappx(cell, lmbda=params.lmbda_fa, iter_solve=params.fa_iter_solve)

cell_ref = deepcopy(cell)

indices = []
modes_cnt = 0
fq_cpy = np.copy(ampls_fa)
while modes_cnt < params.n_modes:
    max_idx = np.unravel_index(fq_cpy.argmax(), fq_cpy.shape)

    indices.append(max_idx)
    fq_cpy[max_idx] = 0.0
    modes_cnt += 1

k_idxs = [pair[1] for pair in indices]
l_idxs = [pair[0] for pair in indices]

###########################################
# cell_01
###########################################

second_guess = interface.get_pmf(nhi, nhj, U, V)

triangle = utils.gen_triangle(lon_v, lat_v)
cell.get_masked(triangle=triangle)

if params.dfft_first_guess:
    second_guess.fobj.set_kls(k_idxs, l_idxs, recompute_nhij = True, components='real')
else:
    second_guess.fobj.set_kls(k_idxs, l_idxs, recompute_nhij = False)

ampls_01, uw_01, dat_2D_01 = second_guess.sappx(cell, lmbda=params.lmbda_sa, updt_analysis=True, scale=1.0, iter_solve=params.sa_iter_solve, save_am=True)

if do_rhs_recomputation:
    ampls_01_rc, uw_01_rc, dat_2D_01_rc = second_guess.recompute_rhs(cell_ref, second_guess.fobj, save_coeffs = True)

###########################################
# cell_02
###########################################

cell_02 = var.topo_cell()
cell_02.topo = total

if delaunay_decomposition:
    vid = utils.delaunay(grid, cell_02, ymax = 2.0*np.pi*scale_fac, xmax = 2.0*np.pi*scale_fac, res_x=res_x, res_y=res_y, tri='upper')
else:
    vid = utils.isosceles(grid, cell_02, ymax = 2.0*np.pi*scale_fac, xmax = 2.0 * np.pi*scale_fac, res=res_x, tri='left')

lat_v = grid.clat_vertices[vid,:]
lon_v = grid.clon_vertices[vid,:]

cell_02.gen_mgrids()

triangle = utils.gen_triangle(lon_v, lat_v, x_rng=[cell_02.lon.min(),cell_02.lon.max()], y_rng=[cell_02.lat.min(),cell_02.lat.max()])
# cell_02.get_masked(mask=np.ones_like(cell_02.topo).astype('bool'))
cell_02.get_masked(triangle=triangle)
cell_02.wlat = np.diff(cell_02.lat).mean()
cell_02.wlon = np.diff(cell_02.lon).mean()

if not delaunay_decomposition:
    second_guess = interface.get_pmf( int(nhi/2), nhj, U, V)

    valid = np.where(np.array(k_idxs) < int(nhi/2))
    k_idxs = np.array(k_idxs)[valid]
    l_idxs = np.array(l_idxs)[valid]

    k_idxs = k_idxs[:int(params.n_modes/2)]
    l_idxs = l_idxs[:int(params.n_modes/2)]

if params.dfft_first_guess:
    second_guess.fobj.set_kls(k_idxs, l_idxs, recompute_nhij = True, components='real')
else:
    second_guess.fobj.set_kls(k_idxs, l_idxs, recompute_nhij = False)

ampls_02, uw_02, dat_2D_02 = second_guess.sappx(cell_02, lmbda=params.lmbda_sa, updt_analysis=True, scale=1.0, iter_solve=params.sa_iter_solve, save_am=True)

if do_rhs_recomputation:
    ampls_02_rc, uw_02_rc, dat_2D_02_rc = second_guess.recompute_rhs(cell_ref, second_guess.fobj, save_coeffs = True)

###########################################
# cell_03
###########################################
if not delaunay_decomposition:
    cell_03 = var.topo_cell()
    cell_03.topo = total

    vid = utils.isosceles(grid, cell_03, ymax = 2.0*np.pi*scale_fac, xmax = 2.0 * np.pi*scale_fac, res=res_x, tri='right')

    # vid = utils.isosceles(grid, cell_03, ymax = 2.0*np.pi*scale_fac, xmax = 2.0 * np.pi*scale_fac, res=res_x, tri='right')

    lat_v = grid.clat_vertices[vid,:]
    lon_v = grid.clon_vertices[vid,:]

    cell_03.gen_mgrids()

    triangle = utils.gen_triangle(lon_v, lat_v, x_rng=[cell_03.lon.min(),cell_03.lon.max()], y_rng=[cell_03.lat.min(),cell_03.lat.max()])
    # cell_03.get_masked(mask=np.ones_like(cell_03.topo).astype('bool'))
    cell_03.get_masked(triangle=triangle)
    cell_03.wlat = np.diff(cell_03.lat).mean()
    cell_03.wlon = np.diff(cell_03.lon).mean()

    ampls_03, uw_03, dat_2D_03 = second_guess.sappx(cell_03, lmbda=params.lmbda_sa, updt_analysis=True, scale=1.0, iter_solve=params.sa_iter_solve, save_am=True)

    if do_rhs_recomputation:
        ampls_03_rc, uw_03_rc, dat_2D_03_rc = second_guess.recompute_rhs(cell_ref, second_guess.fobj, save_coeffs = True)


# %%
sols_fa = (cell, ampls_fa, uw_fa, dat_2D_fa)
params = var.params()
params.plot = True
dplot = diagnostics.diag_plotter(params, nhi, nhj)
dplot.show((0,1), sols_fa, v_extent = v_extent)
print(uw_fa.sum())

sols_01 = (cell, ampls_01, uw_01, dat_2D_01)
params = var.params()
params.plot = True
dplot = diagnostics.diag_plotter(params, nhi, nhj)
dplot.show((0,1), sols_01, v_extent = v_extent)
print(uw_01.sum())

if do_rhs_recomputation:
    sols_01_rc = (cell, ampls_01_rc, uw_01_rc, dat_2D_01_rc)
    params = var.params()
    params.plot = True
    dplot = diagnostics.diag_plotter(params, nhi, nhj)
    dplot.show((0,1), sols_01_rc, v_extent = v_extent)
    print(uw_01_rc.sum())

sols_02 = (cell_02, ampls_02, uw_02, dat_2D_02)
params = var.params()
params.plot = True
dplot = diagnostics.diag_plotter(params, nhi, nhj)
dplot.show((0,1), sols_02, v_extent = v_extent)
print(uw_02.sum())

if do_rhs_recomputation:
    sols_02_rc = (cell, ampls_02_rc, uw_02_rc, dat_2D_02_rc)
    params = var.params()
    params.plot = True
    dplot = diagnostics.diag_plotter(params, nhi, nhj)
    dplot.show((0,1), sols_02_rc, v_extent = v_extent)
    print(uw_02_rc.sum())

if not delaunay_decomposition:
    sols_03 = (cell_03, ampls_03, uw_03, dat_2D_03)
    params = var.params()
    params.plot = True
    dplot = diagnostics.diag_plotter(params, nhi, nhj)
    dplot.show((0,1), sols_03, v_extent = v_extent)
    print(uw_03.sum())

    if do_rhs_recomputation:
        sols_03_rc = (cell, ampls_03_rc, uw_03_rc, dat_2D_03_rc)
        params = var.params()
        params.plot = True
        dplot = diagnostics.diag_plotter(params, nhi, nhj)
        dplot.show((0,1), sols_03_rc, v_extent = v_extent)
        print(uw_03_rc.sum())


print("")

print(ampls_ref.max())
print(ampls_fa.max())
print(ampls_01.max())
print(ampls_02.max())

if delaunay_decomposition:
    print(uw_ref.sum(), uw_01.sum()+uw_02.sum(), (uw_01.sum()+uw_02.sum()) / uw_fa.sum() - 1.0)
else:
    print(ampls_03.max())
    # print(uw_fa.sum(), uw_01.sum()+0.5*uw_02.sum()+0.5*uw_03.sum())
    print(uw_ref.sum(), uw_01.sum()+uw_02.sum()+uw_03.sum())
# %%
