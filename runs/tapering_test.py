# %%
import sys

# setting path
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt

from src import io, var, utils, delaunay
from vis import cart_plot, plotter

from copy import deepcopy

from IPython import get_ipython

ipython = get_ipython()

if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")


def autoreload():
    if ipython is not None:
        ipython.run_line_magic("autoreload", "2")


autoreload()

# %%
# initialise data objects
grid = var.grid()
topo = var.topo_cell()

# we only keep the topography that is inside this lat-lon extent.
params = var.params()

params.merit_cg = 10
params.merit_path = "/home/ray/Documents/orog_data/MERIT/"

params.lat_extent = [48.0, 64.0, 64.0]
params.lon_extent = [-148.0, -148.0, -112.0]

# corresponds to approx (160x160)km
params.delaunay_xnp = 14
params.delaunay_ynp = 11

params.padding = 10

# read grid
reader = io.ncdata(padding=params.padding, padding_tol=(60 - params.padding))
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
# 14x11
tri = delaunay.get_decomposition(
    topo, xnp=params.delaunay_xnp, ynp=params.delaunay_ynp, padding=reader.padding
)
lxkm, lykm = 160, 160
rect_set = np.array([158])

print("rect_set = ", rect_set)

levels = np.linspace(-1000.0, 3000.0, 5)
cart_plot.lat_lon_delaunay(
    topo,
    tri,
    levels,
    label_idxs=True,
    fs=(10, 6),
    highlight_indices=rect_set,
    output_fig=True,
)

# %%
idx = rect_set[0]
cell = var.topo_cell()

rect = False

print("computing idx:", idx)

simplex_lat = tri.tri_lat_verts[idx]
simplex_lon = tri.tri_lon_verts[idx]

utils.get_lat_lon_segments(
    simplex_lat, simplex_lon, cell, topo, rect=rect, load_topo=True, filtered=True
)

cell_orig = deepcopy(cell)

p_length = 20

taper = utils.taper(cell, p_length, art_it=40)
taper.do_tapering()

utils.get_lat_lon_segments(
    simplex_lat,
    simplex_lon,
    cell,
    topo,
    rect=rect,
    padding=p_length,
    load_topo=True,
    filtered=True,
)

utils.get_lat_lon_segments(
    simplex_lat,
    simplex_lon,
    cell,
    topo,
    rect=rect,
    padding=p_length,
    topo_mask=taper.p,
    mask=(taper.p > 1e-2).astype(bool),
    filtered=False,
)

test = cell.topo

# %%
autoreload()
fig_3d = plotter.plot_3d(cell)

p_topo = np.pad(cell_orig.topo, (p_length, p_length), mode="constant")
p_mask = np.pad(cell_orig.mask, (p_length, p_length), mode="constant")
Z = p_topo * p_mask

fig_3d.plot(Z, output_fn="before_taper")
fig_3d.plot(cell.topo, output_fn="after_taper")

lbls = ["longitude [km]", "latitude [km]", "mask"]

fig_3d.plot(p_mask, output_fn="mask_before_taper", lbls=lbls)
fig_3d.plot(taper.p, output_fn="mask_after_taper", lbls=lbls)

# %%

x = np.arange(taper.p.shape[0])
y = np.arange(taper.p.shape[1])

plt.figure()
plt.imshow((p_topo * p_mask) - cell.topo)
plt.show()
# %%

plt.figure()
plt.imshow(cell.topo)
plt.show()
# %%
