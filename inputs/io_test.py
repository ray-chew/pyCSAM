# %%
import sys
import os
# set system path to find local modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import io, var, utils, fourier, physics, delaunay
from wrappers import interface
from vis import plotter, cart_plot

import h5py

# %%
fn = '../outputs/backup/lam_alaska_merit_191023_151813.h5'
file = h5py.File(fn)

reader = io.reader(fn)

params = var.obj()
reader.get_params(params)

for idx in params.rect_set:
    cell = var.topo_cell()

    reader.read_all(idx, cell)

    fs = (15,5.0)
    fig, axs = plt.subplots(1,3, figsize=fs)
    fig_obj = plotter.fig_obj(fig, params.nhi, params.nhj)
    axs[0] = fig_obj.phys_panel(axs[0], cell.data, title='T%i + T%i: FFT reconstruction' %(idx-1, idx), xlabel='longitude [km]', ylabel='latitude [km]', extent=[cell.lon.min(), cell.lon.max(), cell.lat.min(), cell.lat.max()])

    axs[1] = fig_obj.freq_panel(axs[1], cell.spec)
    axs[2] = fig_obj.freq_panel(axs[2], cell.pmf, title="PMF spectrum")
    plt.tight_layout()
    plt.show()

file.close()

# %%
