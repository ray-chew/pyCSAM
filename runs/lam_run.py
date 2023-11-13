import numpy as np
from src import var

params = var.params()

dfft_fa = False
coarse = False

dfft_tag = 'dfft' if dfft_fa else 'lsff'
coarse_tag = 'R2B4' if coarse else 'R2B5'

params.fn_tag = 'lam_alaska_%s_fa_%s' %(dfft_tag, coarse_tag)

if dfft_fa:
    params.dfft_first_guess = True
else:
    params.dfft_first_guess = False

params.lat_extent = [48.,64.,64.]
params.lon_extent = [-148.,-148.,-112.]

params.get_delaunay_triangulation = True
# corresponds to approx (160x160)km

if coarse:
    params.delaunay_xnp = 14
    params.delaunay_ynp = 11

    params.lxkm, params.lykm = 160, 160

    params.nhi = 32
    params.nhj = 64
    params.n_modes = 100
else:
    params.delaunay_xnp = 28
    params.delaunay_ynp = 22

    params.lxkm, params.lykm = 80, 80

    params.nhi = 16
    params.nhj = 32
    params.n_modes = 50

params.rect_set = np.sort([0,1,2,3])

params.lmbda_fa = 1e-1 # first guess
params.lmbda_sa = 1e-1 # second step

params.U, params.V = 10.0, 0.0

params.run_full_land_model = True

params.padding = 10
params.taper_ref = True
params.taper_fa = True
params.taper_sa = True
params.taper_art_it = 20