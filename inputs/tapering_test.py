# %%
import sys
# setting path
sys.path.append('..')

import matplotlib
matplotlib.use('Qt5Agg')
%matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import io, var, utils, fourier, lin_reg, reconstruction, physics, delaunay
from runs import interface
from vis import plotter, cart_plot

import importlib
importlib.reload(io)
importlib.reload(var)
importlib.reload(utils)
importlib.reload(fourier)
importlib.reload(lin_reg)
importlib.reload(reconstruction)
importlib.reload(physics)
importlib.reload(delaunay)

importlib.reload(interface)

importlib.reload(plotter)
importlib.reload(cart_plot)

# %%
# initialise data objects
grid = var.grid()
topo = var.topo_cell()

# read grid
reader = io.ncdata()
fn = '../data/icon_compact.nc'
reader.read_dat(fn, grid)
grid.apply_f(utils.rad2deg) 

# read topography
fn = '../data/topo_compact.nc'
reader.read_dat(fn, topo)

# we only keep the topography that is inside this lat-lon extent.
lat_verts = np.array([52.,64.,64.])
lon_verts = np.array([-141.,-158.,-127.])

reader.read_topo(topo, topo, lon_verts, lat_verts)

# path = "/home/ray/Documents/orog_data/MERIT/"
# reader.read_merit_topo(topo, path, lat_verts, lon_verts)

topo.gen_mgrids()

# Plot the loaded topography...
cart_plot.lat_lon(topo, int=1)

# %%
# Setup Delaunay triangulation domain.

#8x6
# tri = delaunay.get_decomposition(topo, xnp=8, ynp=6)
# rect_set = np.sort([24,62,32,40,66,68,48,30,0,12])
# rect_set = np.sort([0,48,68])
# rect_set = np.sort([40,48])

#11x6
# rect_set = np.sort([0,4,54,92,52,16,44,48,88,58,94])

# 11x9
# tri = delaunay.get_decomposition(topo, xnp=11, ynp=9)
# rect_set = np.sort([36,58,74,118,24,54,98,130,102,34])
# lxkm, lykm = 160, 160

#11x11
# tri = delaunay.get_decomposition(topo, xnp=11, ynp=11)
# rect_set = np.sort([20,26,4,74,130,168,180,102,40,112])
# lxkm, lykm = 160, 120
# rect_set = np.array([102])

#16x11
tri = delaunay.get_decomposition(topo, xnp=16, ynp=11)
rect_set = np.sort([156,154,32,72,68,160,96,162,276,60])
lxkm, lykm = 120, 120
# rect_set = np.array([68])

#22x22
# tri = delaunay.get_decomposition(topo, xnp=22, ynp=22)
# rect_set = np.sort([88,320,126,392,714,262,732,784,112])
# lxkm, lykm = 85, 60
# # rect_set = np.array([392])

print("rect_set = ", rect_set)

levels = np.linspace(-1000.0, 3000.0, 5)
cart_plot.lat_lon_delaunay(topo, tri, levels, label_idxs=True, fs=(10,6), highlight_indices=rect_set, output_fig=True)

# %%
idx = rect_set[0]
cell = var.topo_cell()

rect = False

print("computing idx:", idx)

simplex_lat = tri.tri_lat_verts[idx]
simplex_lon = tri.tri_lon_verts[idx]

triangle = utils.triangle(simplex_lon, simplex_lat)
utils.get_lat_lon_segments(tri.tri_lat_verts[idx], tri.tri_lon_verts[idx], cell, topo, triangle, rect=rect)

taper = utils.taper(cell, 50, art_it=1000)
taper.do_tapering()

utils.get_lat_lon_segments(tri.tri_lat_verts[idx], tri.tri_lon_verts[idx], cell, topo, triangle, rect=rect, padding=50)

test = cell.topo * taper.p


# %%
from matplotlib import cm
from matplotlib.ticker import LinearLocator
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
x = np.arange(taper.p.shape[0])
y = np.arange(taper.p.shape[1])
X,Y = np.meshgrid(y,x)
# Z = taper.p
Z = cell.topo * cell.mask
Z = test

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()


# %%
from matplotlib import cm
from matplotlib.ticker import LinearLocator
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
x = np.arange(taper.p.shape[0])
y = np.arange(taper.p.shape[1])
X,Y = np.meshgrid(y,x)
Z = taper.p

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

# %%