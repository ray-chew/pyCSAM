import numpy as np
from src import var

params = var.obj()

params.fn_grid = '/home/ray/git-projects/spec_appx/data/icon_compact.nc'
params.fn_topo = '/home/ray/git-projects/spec_appx/data/topo_compact.nc'
params.output_fn = 'test_selected'

params.lat_extent = [52.,64.,64.]
params.lon_extent = [-141.,-158.,-127.]

params.delaunay_xnp = 16
params.delaunay_ynp = 11
params.rect_set = np.sort([156,154,32,72,68,160,96,162,276,60])
# rect_set = np.sort([156,154,32,72,160,96,162,276])
# rect_set = np.sort([52,62,110,280,296,298,178,276,244,242])
# rect_set = np.sort([276])
params.lxkm, params.lykm = 120, 120

# Setup the Fourier parameters and object.
params.nhi = 24
params.nhj = 48

params.n_modes = 100

params.U, params.V = 10.0, 0.1

params.cg_spsp = False # coarse grain the spectral space?
params.rect = False if params.cg_spsp else True 

params.tapering        = True
params.taper_first     = False
params.taper_full_fg   = False
params.taper_second    = True
params.taper_both      = False

params.rect = True
params.padding = 50

params.debug = False
params.debug_writer = True
params.dfft_first_guess = False
params.refine = False
params.verbose = False

params.plot = False