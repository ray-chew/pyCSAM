# %%
import sys
# setting path
sys.path.append('..')

import numpy as np
import pandas as pdlot
import matplotlib.pyplot as plt

from src import io, var, utils, delaunay
from vis import cart_plot

from copy import deepcopy

from IPython import get_ipython
ipython = get_ipython()

if ipython is not None:
    ipython.run_line_magic('load_ext', 'autoreload')

def autoreload():
    if ipython is not None:
        ipython.run_line_magic('autoreload', '2')

autoreload()

# %%
# initialise data objects
grid = var.grid()
topo = var.topo_cell()

# we only keep the topography that is inside this lat-lon extent.
params = var.params()

params.merit_cg = 10
params.merit_path = '/home/ray/Documents/orog_data/MERIT/'

params.lat_extent = [48.,64.,64.]
params.lon_extent = [-148.,-148.,-112.]

# corresponds to approx (160x160)km
params.delaunay_xnp = 14
params.delaunay_ynp = 11

params.padding = 10

# read grid
reader = io.ncdata(padding=params.padding, padding_tol=(60-params.padding))
reader.read_dat(params.fn_grid, grid)
grid.apply_f(utils.rad2deg) 

# # read topography
# fn = '../data/topo_compact.nc'
# reader.read_dat(fn, topo)

# reader.read_topo(topo, topo, lon_verts, lat_verts)
reader.read_merit_topo(topo, params)
topo.topo[np.where(topo.topo < -500.0)] = -500.0

topo.gen_mgrids()

# Plot the loaded topography...
cart_plot.lat_lon(topo, int=1)

# %%
# Setup Delaunay triangulation domain.
#14x11
tri = delaunay.get_decomposition(topo, xnp=params.delaunay_xnp, ynp=params.delaunay_ynp, padding = reader.padding)
lxkm, lykm = 160, 160
rect_set = np.array([158])

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

utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell, topo, rect=rect, load_topo=True, filtered=True)

cell_orig = deepcopy(cell)

p_length = 20

taper = utils.taper(cell, p_length, art_it=40)
taper.do_tapering()

utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell, topo, rect=rect, padding=p_length, load_topo=True, filtered=True)

utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell, topo, rect=rect, padding=p_length, topo_mask=taper.p, mask=(taper.p > 1e-2).astype(bool), filtered=False)

test = cell.topo

# %%
ele = 5
azi = 230
cpad = 0.01

plt.rcParams.update({'font.size': 15})
from matplotlib import cm
from matplotlib.ticker import LinearLocator
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10,10))

# Make data.
x = cell.lon / 1000.0
y = cell.lat / 1000.0
X,Y = np.meshgrid(x,y)

p_topo = np.pad(cell_orig.topo, (p_length,p_length), mode='constant')
p_mask = np.pad(cell_orig.mask, (p_length,p_length), mode='constant')
Z = p_topo * p_mask

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.4, pad=cpad)
ax.view_init(ele, azi)
ax.set_xlabel( "longitude [km]", labelpad=10)
ax.set_ylabel( "latitude [km]" , labelpad=10)
ax.set_zlabel( "elevation [m]")
# ax.set_title("orography before tapering")

for label in ax.yaxis.get_ticklabels()[0::2]:
    label.set_visible(False)

plt.tight_layout()
plt.savefig("../manuscript/before_taper.pdf", dpi=200, bbox_inches="tight")
plt.show()


# %%
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10,10))

# Make data.
x = cell.lon / 1000.0
y = cell.lat / 1000.0
X,Y = np.meshgrid(x,y)
Z = cell.topo

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.4, pad=cpad)
ax.view_init(ele, azi)
ax.set_xlabel( "longitude [km]", labelpad=10)
ax.set_ylabel( "latitude [km]" , labelpad=10)
ax.set_zlabel( "elevation [m]")

for label in ax.yaxis.get_ticklabels()[0::2]:
    label.set_visible(False)

plt.tight_layout()
plt.savefig("../manuscript/after_taper.pdf", dpi=200, bbox_inches="tight")
plt.show()

# %%
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10,10))

# Make data.
x = cell.lon / 1000.0
y = cell.lat / 1000.0
X,Y = np.meshgrid(x,y)
Z = p_mask

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.4, pad=cpad)
ax.view_init(ele, azi)
ax.set_xlabel( "longitude [km]", labelpad=10)
ax.set_ylabel( "latitude [km]" , labelpad=10)
ax.set_zlabel( "mask", rotation=90)

for label in ax.yaxis.get_ticklabels()[0::2]:
    label.set_visible(False)

plt.tight_layout()
plt.savefig("../manuscript/mask_before_taper.pdf", dpi=200, bbox_inches="tight")
plt.show()

# %%
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10,10))

# Make data.
x = cell.lon / 1000.0
y = cell.lat / 1000.0
X,Y = np.meshgrid(x,y)
Z = taper.p

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.4, pad=cpad)
ax.view_init(ele, azi)
ax.set_xlabel( "longitude [km]", labelpad=10)
ax.set_ylabel( "latitude [km]" , labelpad=10)
ax.set_zlabel( "mask", rotation=90)


for label in ax.yaxis.get_ticklabels()[0::2]:
    label.set_visible(False)

plt.tight_layout()
plt.savefig("../manuscript/mask_after_taper.pdf", dpi=200, bbox_inches="tight")
plt.show()

# %%

x = np.arange(taper.p.shape[0])
y = np.arange(taper.p.shape[1])

plt.figure()
plt.imshow( (p_topo * p_mask) - cell.topo)
plt.show()
# %%

plt.figure()
plt.imshow(cell.topo)
plt.show()
# %%
