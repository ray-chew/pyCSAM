import numpy as np
from src import var

params = var.params()

params.fn_tag = 'lam_alaska'

params.lat_extent = [52.,64.,64.]
params.lon_extent = [-141.,-158.,-127.]

params.get_delaunay_triangulation = True
params.delaunay_xnp = 16
params.delaunay_ynp = 11
# params.rect_set = np.sort([156,154,32,72,68,160,96,162,276,60])

params.rect_set = np.sort([0, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 32, 34, 36, 38, 42, 44, 58, 60, 66, 68, 70, 72, 74, 76, 78, 80, 96, 98, 100, 102, 108, 110, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 150, 154, 156, 158, 160, 162, 168, 170, 172, 178, 180, 182, 184, 186, 188, 190, 192, 194, 216, 218, 224, 226, 228, 230, 232, 234, 244, 246, 250, 252, 254, 256, 258, 260, 264, 266, 272, 274, 276, 278, 280, 282, 294, 298])
# rect_set = np.sort([156,154,32,72,160,96,162,276])
# rect_set = np.sort([52,62,110,280,296,298,178,276,244,242])
# rect_set = np.sort([276])
params.lxkm, params.lykm = 120, 120

params.run_full_land_model = True