# %%
import sys
import os
# set system path to find local modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import importlib

from src import io, var
from vis import plotter, cart_plot

import h5py

# %%
fn = '../outputs/backup/lam_alaska_merit_191023_151813.h5'
file = h5py.File(fn)

reader = io.reader(fn)

params = var.obj()
reader.get_params(params)

pmf_refs = []
pmf_diffs = []
for rect_idx in params.rect_set:
    pmf_ref = reader.read_data(rect_idx, 'pmf_ref').sum()
    pmf_refs.append(pmf_ref)

    pmfs = []
    for idx in range(rect_idx, rect_idx+2):

        pmf = reader.read_data(idx, 'pmf_sg').sum()
        pmfs.append(pmf)

    assert len(pmfs) == 2, "incorrect append to pmf length"

    pmf_avg = np.array(pmfs).mean()
    pmf_diffs.append(pmf_avg - pmf_ref)
    
file.close()

# %%
pmf_refs = np.array(pmf_refs)
pmf_diffs = np.array(pmf_diffs)
print(pmf_refs.max())

print(pmf_refs.shape)
print(pmf_diffs.shape)

pmf_percent_diff = pmf_diffs / np.abs(pmf_refs).max()
pmf_percent_diff *= 100

plotter.error_bar_plot(params.rect_set, pmf_percent_diff, params, gen_title=True, ylim=[-50,50])


# %%
lat_verts = np.array(params.lat_extent)
lon_verts = np.array(params.lon_extent)

reader = io.ncdata(padding=params.padding, padding_tol=(60-params.padding))

# read topography
topo = var.topo_cell()
if not params.enable_merit:
    reader.read_dat(params.fn_topo, topo)
    reader.read_topo(topo, topo, lon_verts, lat_verts)
else:
    reader.read_merit_topo(topo, params)
    topo.topo[np.where(topo.topo < -500.0)] = -500.0

topo.gen_mgrids()

# %%
# Plot the loaded topography...
importlib.reload(cart_plot)
cart_plot.lat_lon(topo, int=1)


# %%
importlib.reload(cart_plot)
tri = var.obj()

file = h5py.File(fn)
reader = io.reader(fn)

tri.simplices = reader.read_data('decomposition', 'simplices')
tri.points = reader.read_data('decomposition', 'points')
tri.tri_clats = reader.read_data('decomposition', 'tri_clats')
tri.tri_clons = reader.read_data('decomposition', 'tri_clons')
   
file.close()

errors = np.zeros((len(tri.simplices)))
errors[:] = np.nan
errors[params.rect_set] = pmf_percent_diff
errors[np.array(params.rect_set)+1] = pmf_percent_diff

levels = np.linspace(-1000.0, 3000.0, 5)
cart_plot.error_delaunay(topo, tri, label_idxs=True, fs=(15,10), highlight_indices=params.rect_set, output_fig=False, iint=1, errors=errors, alpha_max=0.6)

# %%
