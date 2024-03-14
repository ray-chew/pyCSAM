import numpy as np
from src import var

params = var.params()

# params.output_fn = 'iterative_run_selected_cells'
# params.output_fn = 'test_selected_lsff'

run_case = "POT_BIAS"
run_case = "ITER_REF"

if run_case == "POT_BIAS":
    # potential biases study
    params.rect_set = np.sort([24,200])
    params.no_corrections = True
    params.plot = True

    params.nhi = 32
    params.nhj = 64

elif run_case == "ITER_REF":

    params.plot = True
    params.no_corrections = False
    params.ir_plot_titles = True

    params.nhi = 16
    params.nhj = 32

    # iterative refinement: worst offenders
    params.rect_set = np.sort([92,24, 152,160,42,200,202,238,180])

    # iterative refinement: focus
    # params.rect_set = np.sort([42])


if len(run_case) > 0:
    suffix_tag = '_' + run_case

dfft_fa = False
dfft_tag = 'dfft' if dfft_fa else 'lsff'
params.run_case = run_case
params.fn_tag = 'selected_alaska%s_%s_fa' %(suffix_tag, dfft_tag)

params.lat_extent = [48.,64.,64.]
params.lon_extent = [-148.,-148.,-112.]

# corresponds to approx (160x160)km
params.delaunay_xnp = 14
params.delaunay_ynp = 11

# (xnp x ynp) = (14 x 11); (16, 11)
# FA dfft vs lsff comparison
# params.rect_set = np.sort([20,148,160,212,38,242,188,176,208,248])
# params.rect_set = np.sort([148,38,242])

# params.nhi = 12
# params.nhj = 24
params.n_modes = 100

# (xnp x ynp) = (16 x 14)
# params.rect_set = np.sort([20,148,160,212,256,242])

# corresponds to approx (80x80)km
# params.delaunay_xnp = 28
# params.delaunay_ynp = 22
# params.rect_set = np.sort([20,148,160,678,312,698, 342])
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

params.run_full_land_model = False

params.dfft_first_guess = False

params.padding = 10
params.taper_ref = True
params.taper_fa = True
params.taper_sa = True
params.taper_art_it = 20

params.fa_iter_solve = True
params.sa_iter_solve = True

params.self_test()