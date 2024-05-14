"""
User input data for the run script :mod:`runs.delaunay_runs` for the studies:
    * Potential Biases (``POT_BIAS``)
    * Iterative refinement (``ITER_REF``)
    * FFT vs LSFF in the First Approximation step (``DFFT_FA`` and ``LSFF_FA``)
    * Complementary study on the flux computation; does not appear in the manuscript (``FLUX_SDY``) 
"""

import numpy as np
from src import var, utils
from inputs import local_paths

params = var.params()
utils.transfer_attributes(params, local_paths.paths, prefix="path")

# potential biases study
# run_case = "POT_BIAS"
# iterative refinement study
run_case = "ITER_REF"
# DFFT FA run for DFFT vs LSFF comparison
# run_case = "DFFT_FA"
# LSFF FA run for DFFT vs LSFF comparison
# run_case = "LSFF_FA"
# effective flux contribution study
run_case = "FLUX_SDY"

if run_case == "POT_BIAS":
    params.rect_set = np.sort([24, 200])
    params.no_corrections = True
    params.plot = True

    params.dfft_first_guess = False
    params.nhi = 32
    params.nhj = 64

elif run_case == "ITER_REF":
    params.plot = True
    params.no_corrections = False
    params.ir_plot_titles = True

    params.dfft_first_guess = False
    params.nhi = 16
    params.nhj = 32

    # iterative refinement: worst offenders
    params.rect_set = np.sort([92, 24, 152, 160, 42, 200, 202, 238, 180])
    # iterative refinement: focus
    # params.rect_set = np.sort([42])

elif run_case == "DFFT_FA":
    # FA dfft vs lsff comparison
    params.rect_set = np.sort([20, 148, 160, 212, 38, 242, 188, 176, 208, 248])
    params.dfft_first_guess = True
    params.nhi = 32
    params.nhj = 64
    params.no_corrections = True

elif run_case == "LSFF_FA":
    # FA dfft vs lsff comparison
    params.rect_set = np.sort([20, 148, 160, 212, 38, 242, 188, 176, 208, 248])
    params.dfft_first_guess = False
    params.nhi = 32
    params.nhj = 64
    params.no_corrections = True

elif run_case == "FLUX_SDY":
    params.no_corrections = True
    params.dfft_first_guess = False
    params.nhi = 32
    params.nhj = 64
    params.rect_set = np.sort([158])

    params.recompute_rhs = True
    params.plot = True

else:
    assert 0


if len(run_case) > 0:
    suffix_tag = "_" + run_case

dfft_tag = "dfft" if params.dfft_first_guess else "lsff"
params.run_case = run_case
params.fn_tag = "selected_alaska%s_%s_fa" % (suffix_tag, dfft_tag)

params.lat_extent = [48.0, 64.0, 64.0]
params.lon_extent = [-148.0, -148.0, -112.0]

# corresponds to approx (160x160)km
params.delaunay_xnp = 14
params.delaunay_ynp = 11

params.n_modes = 100

params.lmbda_fa = 1e-1  # first guess
params.lmbda_sa = 1e-1  # second step

params.lxkm, params.lykm = 160, 160

params.U, params.V = 10.0, 0.0

params.run_full_land_model = False

params.padding = 10
params.taper_ref = True
params.taper_fa = True
params.taper_sa = True
params.taper_art_it = 20

params.fa_iter_solve = True
params.sa_iter_solve = True

params.self_test()
