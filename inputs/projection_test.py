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

%load_ext autoreload
%autoreload

# %%
# from inputs.lam_run import params
from inputs.selected_run import params
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
%autoreload
cart_plot.lat_lon(topo, int=1)

levels = np.linspace(-500.0, 3000.0, 8)
cart_plot.lat_lon_delaunay(topo, tri, levels, label_idxs=True, fs=(15,10), highlight_indices=params.rect_set, output_fig=False, int=1)

print(topo.lat.shape)
print(topo.lon_grid.shape)

print(np.diff(topo.lat).max())
print(np.diff(topo.lat).min())

# %%
import cartopy.crs as ccrs
from cartopy.mpl.ticker import (LongitudeFormatter,
                                LatitudeFormatter,
                                LatitudeLocator, LongitudeLocator)

fs   = (10,6)
iint = 40
jint = 80

fig = plt.figure(figsize=fs)
ax = plt.axes(projection=ccrs.PlateCarree())

ax.coastlines()
im = ax.scatter(topo.lon_grid[::iint,::iint], topo.lat_grid[::iint,::iint],
            alpha=0.5,
            transform=ccrs.PlateCarree(),
            cmap='GnBu',
            s=0.1
            )

cax = fig.add_axes([0.99, 0.22, 0.025, 0.55])
fig.colorbar(im, cax=cax)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.left_labels = False

gl.xlocator = LongitudeLocator()
gl.ylocator = LatitudeLocator()
gl.xformatter = LongitudeFormatter(auto_hide=False)
gl.yformatter = LatitudeFormatter()

ax.text(-0.01, 0.5, 'latitude', va='bottom', ha='center',
        rotation='vertical', rotation_mode='anchor',
        transform=ax.transAxes)
ax.text(0.5, -0.15, 'longitude', va='bottom', ha='center',
        rotation='horizontal', rotation_mode='anchor',
        transform=ax.transAxes)

ax.tick_params(axis="both",
            tickdir='out',
            length=15,
            grid_transform=ccrs.PlateCarree())

plt.show()

# %%