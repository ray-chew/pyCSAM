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

# corresponds to approx (160x160)km
params.delaunay_xnp = 14
params.delaunay_ynp = 11

# (xnp x ynp) = (14 x 11); (16, 11)
params.rect_set = np.sort([20,148,160,212,38,242])

# worst underestimators
params.rect_set = np.sort([152,160,42,200,202,238,180])

# worst overestimators
params.rect_set = np.sort([92,24,28])

# params.rect_set = np.sort([160])

# params.nhi = 32
# params.nhj = 64

params.nhi = 12
params.nhj = 24
params.n_modes = 100

# (xnp x ynp) = (16 x 14)
# params.rect_set = np.sort([20,148,160,212,256,242])

# corresponds to approx (80x80)km
# params.delaunay_xnp = 28
# params.delaunay_ynp = 22
# params.rect_set = np.sort([20,148,160,678,312,698, 342])
# params.nhi = 16
# params.nhj = 32
# params.n_modes = 50
# look into 342!!
# params.rect_set = np.sort([342])
# params.rect_set = np.sort([342,346,652,674,670,348,654,656])


# params.rect_set = np.sort([598,908,906,902,896,898,622,902])

# params.rect_set = np.sort([116])
# all the main MERIT x10 offenders. To test implementation of correction strategy.
# params.rect_set = [20, 66, 182, 240]
# params.rect_set = [182]

# MERIT full LAM top underestimators AFTER correction... Why does correction not work?
# params.rect_set = np.sort([98, 210, 286, 80, 266])

# MERIT full LAM top overestimators AFTER correction
# params.rect_set = np.sort([0, 6, 212, 84, 174])
# params.rect_set = np.sort([212, 174])


params.lmbda_fa = 1e-1 # first guess
params.lmbda_sa = 1e-1 # second step

params.lxkm, params.lykm = 160, 160

params.U, params.V = 10.0, 0.0
# params.V = 0.0

params.run_full_land_model = False

params.padding = 10

params.dfft_first_guess = False

params.taper_ref = True
params.taper_fa = True
params.taper_sa = True
params.taper_art_it = 20

params.no_corrections = False

params.plot = True
