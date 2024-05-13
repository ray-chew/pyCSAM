import numpy as np
from src import var

params = var.params()

params.output_path = "/home/ray/git-projects/spec_appx/outputs/"
params.output_fn = "icon_merit_reg"
params.fn_grid = "../data/icon_compact.nc"
params.fn_topo = "../data/topo_compact.nc"
params.lat_extent = [48.0, 64.0, 64.0]
params.lon_extent = [-148.0, -148.0, -112.0]

params.tri_set = [13, 104, 105, 106]

# Setup the Fourier parameters and object.
params.nhi = 24
params.nhj = 48

params.n_modes = 50

params.U, params.V = 10.0, 0.0

params.rect = True

params.debug = False
params.dfft_first_guess = True
params.refine = False
params.verbose = False

params.plot = True