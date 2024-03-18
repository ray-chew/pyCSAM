"""User-defined parameters used in the debugger
"""

import numpy as np
from src import var

params = var.obj()

params.fn_grid = "/home/ray/git-projects/spec_appx/data/icon_compact.nc"
params.fn_topo = "/home/ray/git-projects/spec_appx/data/topo_compact.nc"
params.output_fn = "debug_run"

params.lat_extent = [52.0, 64.0, 64.0]
params.lon_extent = [-141.0, -158.0, -127.0]

params.delaunay_xnp = 16
params.delaunay_ynp = 11

###############################
# USGS GMTED2010 Alaska-LAM run
###############################

# severe overstimation (>10% error)
params.rect_set = [102, 274, 188, 294, 58, 216, 8, 190, 136, 258]
# >30% error
params.rect_set = [274, 188]
# ~10-20% error
# params.rect_set = [136, 258, 194]
# largest error
# params.rect_set = [102]

# severe underestimation (>17% error)
params.rect_set = [
    246,
    282,
    44,
    22,
    20,
    134,
    234,
    228,
    192,
    230,
    6,
    250,
    168,
    226,
    280,
    244,
    278,
    142,
    66,
    260,
]

# cells with >25% error
# params.rect_set = [246, 282, 44, 22, 20, 134, 234]

# cells with 15%<25% error
params.rect_set = [
    228,
    192,
    230,
    6,
    250,
    168,
    226,
    280,
    244,
    278,
    142,
    66,
    260,
    144,
    172,
    78,
    10,
    154,
    170,
    70,
    140,
]
# smaller subsets
# params.rect_set = [228, 192, 230, 6, 250, 168, 226, 280, 244, 278, 142, 66, 260]
# params.rect_set = [228, 192, 230, 6, 250, 168, 226]
# params.rect_set = [260]


###############################
# MERIT DEM Alaska-LAM run
###############################

params.merit_cg = 10
params.merit_path = "/home/ray/Documents/orog_data/MERIT/"

# +39% overestimation
params.rect_set = [20]

# -23.3%; -22.6%; -15.9% underestimation
params.rect_set = [66, 182, 240]

params.lxkm, params.lykm = 120, 120

# Setup the Fourier parameters and object.
params.nhi = 24
params.nhj = 48

params.n_modes = 100

params.U, params.V = 10.0, 0.1

params.cg_spsp = False  # coarse grain the spectral space?
params.rect = False if params.cg_spsp else True

params.lmbda_fg = 1e-2
params.lmbda_sg = 1e-1

params.tapering = True
params.taper_first = False
params.taper_full_fg = False
params.taper_second = True
params.taper_both = False

params.rect = True
params.padding = 50

params.debug = False
params.debug_writer = True
params.dfft_first_guess = False

params.refine = True
params.refine_n_modes = 50
params.refine_lmbda_fg = 1e-2
params.refine_lmbda_sg = 0.2

params.verbose = False

params.plot = True
