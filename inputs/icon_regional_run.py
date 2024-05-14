import numpy as np
from src import var, utils
from inputs import local_paths

params = var.params()

params.fn_output = "icon_merit_reg"
utils.transfer_attributes(params, local_paths.paths, prefix="path")

print(True)

### alaska
params.lat_extent = [48.0, 64.0, 64.0]
params.lon_extent = [-148.0, -148.0, -112.0]

### Tierra del Fuego
params.lat_extent = [-38.0, -56.0, -56.0]
params.lon_extent = [-76.0, -76.0, -53.0]

### South Pole
params.lat_extent = [-75.0, -61.0, -61.0]
params.lon_extent = [-77.0, -50.0, -50.0]

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