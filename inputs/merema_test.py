# %%
import sys
# setting path
sys.path.append('..')

from src import io, var, utils, fourier, lin_reg, reconstruction, physics, delaunay
from wrappers import interface
from vis import plotter, cart_plot

from netCDF4 import Dataset
import numpy as np

# %%
# we only keep the topography that is inside this lat-lon extent.
# assuming contiguous extent
lat_verts = np.array([52.,64.,64.])
lon_verts = np.array([-141.,-158.,-127.])

dir = "/home/ray/Documents/orog_data/MERIT/"

# %%
def compute_idx(vert, typ, direction):
    if direction == 'lon':
        fn_int = fn_lon
    else:
        fn_int = fn_lat
    
    sgn = np.sign(vert)
    where_idx = np.argmin(np.abs(fn_int - vert))
    print(fn_int, where_idx)
    
    if typ=='min':
        if (vert - fn_int[where_idx]) < 0.0:
            if direction == 'lon':
                where_idx -= 1
            else:
                where_idx += 1
    elif typ == 'max':
        if (vert - fn_int[where_idx]) > 0.0:
            if direction == 'lon':
                where_idx += 1
            else:
                where_idx -= 1
            
    where_idx = int(where_idx)
            
    print("where_idx, vert, fn_int[where_idx] for typ:")
    print(where_idx, vert, fn_int[where_idx], typ)
    print("")
            
    return where_idx

def get_NSEW(vert, typ):
    if typ == 'lat':
        if vert >= 0.0:
            dir_tag = 'N'
        else:
            dir_tag = 'S'
    if typ == 'lon':
        if vert >= 0.0:
            dir_tag = 'E'
        else:
            dir_tag = 'W'
        
    return dir_tag

# %%
fn_lon = np.array([-180.0, -150.0, -120.0, -90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0])
fn_lat = np.array([90.0, 60.0, 30.0, 0.0, -30.0, -60.0, -90.0])

lat_min_idx = compute_idx(lat_verts.min(), 'min', 'lat')
lat_max_idx = compute_idx(lat_verts.max(), 'max', 'lat')

lon_min_idx = compute_idx(lon_verts.min(), 'min', 'lon')
lon_max_idx = compute_idx(lon_verts.max(), 'max', 'lon')


# %%
fns = []

for lat_cnt, lat_idx in enumerate(range(lat_max_idx, lat_min_idx)):
    l_lat_bound, r_lat_bound = fn_lat[lat_idx], fn_lat[lat_idx+1]
    l_lat_tag, r_lat_tag = get_NSEW(l_lat_bound, 'lat'), get_NSEW(r_lat_bound, 'lat')
    
    for lon_cnt, lon_idx in enumerate(range(lon_min_idx, lon_max_idx)):

        l_lon_bound, r_lon_bound = fn_lon[lon_idx], fn_lon[lon_idx+1]
        l_lon_tag, r_lon_tag = get_NSEW(l_lon_bound, 'lon'), get_NSEW(r_lon_bound, 'lon')
            
        name = "MERIT_%s%.2d-%s%.2d_%s%.3d-%s%.3d.nc4" %(l_lat_tag, np.abs(l_lat_bound), r_lat_tag, np.abs(r_lat_bound), l_lon_tag, np.abs(l_lon_bound), r_lon_tag, np.abs(r_lon_bound))

        fns.append(name)

# %%
def get_topo(topos=None):
    populate = False if topos is None else True

    if not populate:
        nc_lon = 0
        nc_lat = 0
    else:
        n_col = 0
        n_row = 0
        lon_sz_old = 0
        lat_sz_old = 0
        lats = []
        lons = []

    for cnt, fn in enumerate(fns):
        test = Dataset(dir+fn)

        lat = test['lat']
        lat_min_idx = np.argmin(np.abs(lat - lat_verts.min()))
        lat_max_idx = np.argmin(np.abs(lat - lat_verts.max()))

        lat_high = np.max((lat_min_idx, lat_max_idx))
        lat_low = np.min((lat_min_idx, lat_max_idx))

        lon = test['lon']
        lon_min_idx = np.argmin(np.abs(lon - (lon_verts.min())))
        lon_max_idx = np.argmin(np.abs(lon - (lon_verts.max())))

        lon_high = np.max((lon_min_idx, lon_max_idx))
        lon_low = np.min((lon_min_idx, lon_max_idx))

        if not populate:
            if cnt < (lon_cnt+1):
                nc_lon += lon_high - lon_low
            if ((cnt % (lat_cnt+1)) == 0):
                nc_lat += lat_high - lat_low
        else:
            topo = test['Elevation'][lat_low:lat_high, lon_low:lon_high]
            if n_col == 0:
                lats += (lat[lat_low:lat_high].tolist())
            if n_row == 0:
                lons += (lon[lon_low:lon_high].tolist())

            lon_sz = lon_high - lon_low
            lat_sz = lat_high - lat_low

            topos[n_row*lat_sz_old:n_row*lat_sz_old+lat_sz, n_col*lon_sz_old:n_col*lon_sz_old+lon_sz] = topo

            n_col += 1
            if n_col == (lon_cnt + 1):
                n_col = 0
                n_row += 1 
                lat_sz_old = np.copy(lat_sz)

            lon_sz_old = np.copy(lon_sz)

        test.close()

    if not populate:
        topos = np.zeros((nc_lat, nc_lon))
        return topos
    else:
        # lats = np.unique(lats)
        # lons = np.unique(lons)

        return np.sort(lats), np.sort(lons)

# %%
topos = get_topo()
topos.shape
lats, lons = get_topo(topos)

print("")


# %%
import importlib
importlib.reload(io)
importlib.reload(cart_plot)

# initialise data objects
grid = var.grid()
topo = var.topo_cell()

# read grid
reader = io.ncdata()
fn = '../data/icon_compact.nc'
reader.read_dat(fn, grid)
grid.apply_f(utils.rad2deg) 

# read topography
fn = '../data/topo_compact.nc'
reader.read_merit_topo(topo, dir, lat_verts, lon_verts)


print("")

# %%
topo.topo[np.where(topo.topo < -500)] = -500
topo.gen_mgrids()

# Plot the loaded topography...
cart_plot.lat_lon(topo, int=200)

# %%
import matplotlib.pyplot as plt

pl_interval = 20

levels = np.linspace(-1000.0, 3000.0, 5)
plt.figure(figsize=(10,6))
plt.contourf(topo.topo.topo[::pl_interval,::pl_interval], levels=levels, origin='upper')
plt.colorbar()
plt.show()
# %%
