# %%
import sys

# set system path to find local modules
sys.path.append("..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import io, var, utils, fourier, physics
from wrappers import interface
from vis import plotter, cart_plot

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
from inputs.icon_regional_run import params

if params.self_test():
    params.print()

grid = var.grid()
topo = var.topo_cell()

# read grid
reader = io.ncdata(padding=params.padding, padding_tol=(60 - params.padding))

# writer object
# writer = io.writer(params.output_fn, params.rect_set, debug=params.debug_writer)

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


# %%

# if params.run_full_land_model:
#     params.rect_set = delaunay.get_land_cells(tri, topo, height_tol=0.5)
#     print(params.rect_set)

# params_orig = deepcopy(params)
# writer.write_all_attrs(params)
# writer.populate("decomposition", "rect_set", params.rect_set)

clon = grid.clon
clat = grid.clat
clon_vertices = grid.clon_vertices
clat_vertices = grid.clat_vertices

ncells, nv = clon_vertices.shape[0], clon_vertices.shape[1]

# -- print information to stdout
print("Cells:            %6d " % clon.size)

# -- create the triangles
clon_vertices = np.where(clon_vertices < -180.0, clon_vertices + 360.0, clon_vertices)
clon_vertices = np.where(clon_vertices > 180.0, clon_vertices - 360.0, clon_vertices)

triangles = np.zeros((ncells, nv, 2), np.float32)

for i in range(0, ncells, 1):
    triangles[i, :, 0] = np.array(clon_vertices[i, :])
    triangles[i, :, 1] = np.array(clat_vertices[i, :])

print("--> triangles done")

cart_plot.lat_lon_icon(topo, triangles, ncells=ncells, clon=clon, clat=clat)

# %%
