"""
User input data for the run script :mod:`runs.delaunay_runs` for the studies:
    * Coarse grid study over the Alaskan Rocky Mountains (``R2B4``)
    * Fine grid study over the Alaskan Rocky Mountains (``R2B5``)
    * Strong wind study over the coarse grid (``R2B4_STRW``)
    * Wind direction studies
"""

import numpy as np
from src import var

params = var.params()

run_case = "R2B4"
# run_case = "R2B5"
# run_case = "R2B4_STRW"

run_case = "R2B4_NN"
run_case = "R2B4_NE"
run_case = "R2B4_SE"
run_case = "R2B4_SS"
run_case = "R2B4_SW"
run_case = "R2B4_WW"
run_case = "R2B4_NW"

if run_case == "R2B4":
    coarse = True
    params.U, params.V = 10.0, 0.0

elif run_case == "R2B5":
    coarse = False
    params.U, params.V = 10.0, 0.0

elif run_case == "R2B4_STRW":
    coarse = True
    params.U, params.V = -40.0, 20.0

elif run_case == "R2B4_NN":
    coarse = True
    params.U, params.V = 0.0, 10.0

elif run_case == "R2B4_NE":
    coarse = True
    params.U, params.V = np.sqrt(50.0), np.sqrt(50.0)

elif run_case == "R2B4_SE":
    coarse = True
    params.U, params.V = np.sqrt(50.0), -np.sqrt(50.0)

elif run_case == "R2B4_SS":
    coarse = True
    params.U, params.V = 0.0, -10.0

elif run_case == "R2B4_SW":
    coarse = True
    params.U, params.V = -np.sqrt(50.0), -np.sqrt(50.0)

elif run_case == "R2B4_WW":
    coarse = True
    params.U, params.V = -10.0, 0.0

elif run_case == "R2B4_NW":
    coarse = True
    params.U, params.V = -np.sqrt(50.0), np.sqrt(50.0)


else:
    assert False

    

if len(run_case) > 0:
    suffix_tag = '_' + run_case

dfft_fa = False
dfft_tag = 'dfft' if dfft_fa else 'lsff'
params.run_case = run_case
params.fn_tag = 'lam_alaska%s_%s_fa' %(suffix_tag, dfft_tag)

if dfft_fa:
    params.dfft_first_guess = True
else:
    params.dfft_first_guess = False

params.lat_extent = [48.,64.,64.]
params.lon_extent = [-148.,-148.,-112.]

params.get_delaunay_triangulation = True

if coarse:
    # corresponds to approx (160x160)km
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

params.run_full_land_model = True

params.padding = 10
params.taper_ref = True
params.taper_fa = True
params.taper_sa = True
params.taper_art_it = 20

params.fa_iter_solve = True
params.sa_iter_solve = True