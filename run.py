import numpy as np
import matplotlib.pyplot as plt

from src import io, var, utils, fourier, lin_reg, reconstruction
from vis import plotter

#### define wavenumber range
nhi = 12
nhj = 12
ll = np.arange(-(nhj/2-1),nhj/2+1)
kk = np.arange(0,nhi)

#### initialise triangle
grid = var.grid()
cell = var.topo_cell()

vid = utils.isoceles(grid, cell)

lat_v = grid.clat_vertices[vid,:]
lon_v = grid.clon_vertices[vid,:]

cell.gen_mgrids()

# define triangle given the vertices
triangle = utils.triangle(lon_v, lat_v)

# get mask array
mask = triangle.vec_get_mask(cell.lon_grid.ravel(), cell.lat_grid.ravel())
mask = mask.reshape(cell.lon_grid.shape)


#### generate reference solution in the triangle
kn, ln = 0.0, 1.0
cell.topo = np.sin( kn * cell.lon_grid + ln * cell.lat_grid)
triangle = utils.triangle(lon_v, lat_v)
cell.get_masked(triangle, mask = mask)

fobj = fourier.f_trans(nhi,nhj)
# fobj.do_full(cell)

# am, data_recons = lin_reg.do(fobj, cell, lmbda = 0.5)

# fobj.get_freq_grid(am)
# dat_2D = reconstruction.recon_2D(data_recons, cell)

# freqs = 2.0 * np.abs(fobj.ampls)

fobj.set_kls([0,1,5,4,9,3,1],[1,0,3,6,7,7,9])
fobj.do_full(cell)

print(fobj.bf_cos)
print(fobj.bf_sin)

am, data_recons = lin_reg.do(fobj, cell, lmbda = 1e-6)

fobj.get_freq_grid(am)
dat_2D = reconstruction.recon_2D(data_recons, cell)

freqs = 1.0 * np.abs(fobj.ampls)