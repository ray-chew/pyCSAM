import numpy as np
from src import var

params = var.params()

params.fn_grid = '/home/ray/git-projects/spec_appx/data/icon_compact.nc'
params.fn_topo = '/home/ray/git-projects/spec_appx/data/topo_compact.nc'

params.merit_cg = 10
params.merit_path = '/home/ray/Documents/orog_data/MERIT/'

params.output_fn = 'test_selected'

params.lat_extent = [52.,64.,64.]
params.lon_extent = [-141.,-158.,-127.]

params.lat_extent = [48.,64.,64.]
params.lon_extent = [-148.,-148.,-112.]

params.delaunay_xnp = 16
params.delaunay_ynp = 11
params.rect_set = np.sort([156,154,32,72,68,160,96,162,276,60])
# rect_set = np.sort([52,62,110,280,296,298,178,276,244,242])
# rect_set = np.sort([276])


# MERIT 8x coarse-graining corresponding selected rect set (approx USGS GMTED2010 resolution).
# params.rect_set = np.sort([188, 204, 280, 102, 78, 162, 160, 146, 106, 164])
params.rect_set = np.sort([156,152,130,78,20,174,176,64, 86, 228])


# MERIT 10x coarse-graining corresponding selected rect set (better approximation of the USGS GMTED2010 resolution).
# PADDING=50
params.rect_set = np.sort([66,182, 20, 216, 246, 244, 240, 152, 278, 22])



params.rect_set = np.sort([168,242,154,162,116,212,20,290,38, 266])
# all the main MERIT x10 offenders. To test implementation of correction strategy.
# params.rect_set = [20, 66, 182, 240]
# params.rect_set = [182]

# MERIT full LAM top underestimators AFTER correction... Why does correction not work?
# params.rect_set = np.sort([98, 210, 286, 80, 266])

# MERIT full LAM top overestimators AFTER correction
# params.rect_set = np.sort([0, 6, 212, 84, 174])
# params.rect_set = np.sort([212, 174])

params.lxkm, params.lykm = 120, 120

params.U, params.V = 40.0, -20.0

params.run_full_land_model = False

params.padding = 0

params.dfft_first_guess = True

params.plot = True
