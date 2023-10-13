import netCDF4 as nc
import numpy as np
import h5py
import os

from src import utils

class ncdata(object):

    def __init__(self, read_merit = False, padding = 0, padding_tol = 10):
        self.read_merit = read_merit
        self.padding = padding_tol + padding

    def read_dat(self, fn, obj):
        df = nc.Dataset(fn)

        for key, _ in vars(obj).items():
            if key in df.variables:
                setattr( obj, key, df.variables[key][:] )

        df.close()

    def get_truths(self, arr, vert_pts, d_pts):
        return (arr >= (vert_pts.min() - self.padding * d_pts)) & (arr <= vert_pts.max() + self.padding * d_pts)

    def read_topo(self, topo, cell, lon_vert, lat_vert):
        lon, lat, z = topo.lon, topo.lat, topo.topo
        
        nrecords = np.shape(z)[0]

        bool_arr = np.zeros_like(z).astype(bool)
        lat_arr = np.zeros_like(z)
        lon_arr = np.zeros_like(z)

        z = z[:,::-1,:]

        for n in range(nrecords):
            lat_n = lat[n]
            lon_n = lon[n]

            dlat, dlon = np.diff(lat_n).mean(), np.diff(lon_n).mean()

            lon_nm, lat_nm = np.meshgrid(lon_n, lat_n)

            bool_arr[n] = self.get_truths(lon_nm, lon_vert, dlon) & self.get_truths(lat_nm, lat_vert, dlat)

            lat_arr[n] = lat_nm
            lon_arr[n] = lon_nm

        lon_res = lon_arr[bool_arr]
        lat_res = lat_arr[bool_arr]
        z_res = z[bool_arr].data


        #---- processing of the lat,lon,topo to get the regular 2D grid for topography
        lon_uniq, lat_uniq = np.unique(lon_res), np.unique(lat_res) # get unique values of lon,lat
        nla = len(lat_uniq); nlo = len(lon_uniq)

        lat_res_sort_idx = np.argsort(lat_res)
        lon_res_sort_idx = np.argsort(lon_res[lat_res_sort_idx].reshape(nla,nlo),axis=1)
        z_res = z_res[lat_res_sort_idx]
        z_res = np.take_along_axis(z_res.reshape(nla,nlo),lon_res_sort_idx,axis=1)
        topo_2D = z_res.reshape(nla,nlo)
    
        print('Data fetched...')
        cell.lon = lon_uniq
        cell.lat = lat_uniq
        cell.topo = topo_2D


    class read_merit_topo(object):
        
        def __init__(self, cell, params):
            self.dir = params.merit_path

            self.fn_lon = np.array([-180.0, -150.0, -120.0, -90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0])
            self.fn_lat = np.array([90.0, 60.0, 30.0, 0.0, -30.0, -60.0, -90.0])

            self.lat_verts = np.array(params.lat_extent)
            self.lon_verts = np.array(params.lon_extent)

            self.merit_cg = params.merit_cg

            lat_min_idx = self.compute_idx(self.lat_verts.min(), 'min', 'lat')
            lat_max_idx = self.compute_idx(self.lat_verts.max(), 'max', 'lat')

            lon_min_idx = self.compute_idx(self.lon_verts.min(), 'min', 'lon')
            lon_max_idx = self.compute_idx(self.lon_verts.max(), 'max', 'lon')

            fns, lon_cnt, lat_cnt = self.get_fns(lat_min_idx, lat_max_idx, lon_min_idx, lon_max_idx)

            self.get_topo(cell, fns, lon_cnt, lat_cnt)


        def compute_idx(self, vert, typ, direction):
            if direction == 'lon':
                fn_int = self.fn_lon
            else:
                fn_int = self.fn_lat

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

        def get_fns(self, lat_min_idx, lat_max_idx, lon_min_idx, lon_max_idx):
            fns = []

            for lat_cnt, lat_idx in enumerate(range(lat_max_idx, lat_min_idx)):
                l_lat_bound, r_lat_bound = self.fn_lat[lat_idx], self.fn_lat[lat_idx+1]
                l_lat_tag, r_lat_tag = self.get_NSEW(l_lat_bound, 'lat'), self.get_NSEW(r_lat_bound, 'lat')
                
                for lon_cnt, lon_idx in enumerate(range(lon_min_idx, lon_max_idx)):

                    l_lon_bound, r_lon_bound = self.fn_lon[lon_idx], self.fn_lon[lon_idx+1]
                    l_lon_tag, r_lon_tag = self.get_NSEW(l_lon_bound, 'lon'), self.get_NSEW(r_lon_bound, 'lon')
                        
                    name = "MERIT_%s%.2d-%s%.2d_%s%.3d-%s%.3d.nc4" %(l_lat_tag, np.abs(l_lat_bound), r_lat_tag, np.abs(r_lat_bound), l_lon_tag, np.abs(l_lon_bound), r_lon_tag, np.abs(r_lon_bound))

                    fns.append(name)

            return fns, lon_cnt, lat_cnt
        

        def get_topo(self, cell, fns, lon_cnt, lat_cnt, init=True, populate=True):
            if (cell.topo is None) and (init): 
                self.get_topo(cell, fns, lon_cnt, lat_cnt, init=False, populate=False)

            if not populate:
                nc_lon = 0
                nc_lat = 0
            else:
                n_col = 0
                n_row = 0
                lon_sz_old = 0
                lat_sz_old = 0
                cell.lat = []
                cell.lon = []

            for cnt, fn in enumerate(fns):
                test = nc.Dataset(self.dir+fn)

                lat = test['lat']
                lat_min_idx = np.argmin(np.abs(lat - self.lat_verts.min()))
                lat_max_idx = np.argmin(np.abs(lat - self.lat_verts.max()))

                lat_high = np.max((lat_min_idx, lat_max_idx))
                lat_low = np.min((lat_min_idx, lat_max_idx))

                lon = test['lon']
                lon_min_idx = np.argmin(np.abs(lon - (self.lon_verts.min())))
                lon_max_idx = np.argmin(np.abs(lon - (self.lon_verts.max())))

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
                        cell.lat += (lat[lat_low:lat_high].tolist())
                    if n_row == 0:
                        cell.lon += (lon[lon_low:lon_high].tolist())

                    lon_sz = lon_high - lon_low
                    lat_sz = lat_high - lat_low

                    cell.topo[n_row*lat_sz_old:n_row*lat_sz_old+lat_sz, n_col*lon_sz_old:n_col*lon_sz_old+lon_sz] = topo

                    n_col += 1
                    if n_col == (lon_cnt + 1):
                        n_col = 0
                        n_row += 1 
                        lat_sz_old = np.copy(lat_sz)

                    lon_sz_old = np.copy(lon_sz)

                test.close()

            if not populate:
                cell.topo = np.zeros((nc_lat, nc_lon))
            else:
                iint = self.merit_cg
                # cell.lat = np.sort(cell.lat)[::iint]
                # cell.lon = np.sort(cell.lon)[::iint][:-1]

                cell.lat = utils.sliding_window_view(np.sort(cell.lat), (iint,), (iint,)).mean(axis=-1)
                cell.lon = utils.sliding_window_view(np.sort(cell.lon), (iint,), (iint,)).mean(axis=-1)


                cell.topo = utils.sliding_window_view(cell.topo, (iint,iint), (iint,iint)).mean(axis=(-1,-2))[::-1,:]

        @staticmethod
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

class writer(object):
    """
    HDF5 writer class. Contains methods to create HDF5 file, create data sets and populate them with output variables.

    """
    def __init__(self, fn, idxs, sfx='', debug=False):
        """
        Creates HDF5 file based on filename given attribute `OUTPUT_FILENAME`.

        """

        self.FORMAT = ".h5"
        self.OUTPUT_FOLDER = "../outputs/"
        self.OUTPUT_FILENAME = fn
        self.OUTPUT_FULLPATH = self.OUTPUT_FOLDER + self.OUTPUT_FILENAME
        self.SUFFIX = sfx
        self.DEBUG = debug
                
        self.IDXS = idxs
        self.PATHS = [  
                        # vars from the 'tri' object
                        'tri_lat_verts',
                        'tri_lon_verts',
                        'tri_clats',
                        'tri_clons',
                        # vars from the 'cell' object
                        'lon',
                        'lat',
                        'lon_grid',
                        'lat_grid',
                        # vars from the 'analysis' object
                        'ampls',
                        'kks',
                        'lls',
                        'recon',
                    ]
        
        self.ATTRS = [
                        # vars from the 'analysis' object
                        'wlat',
                        'wlon',
                    ]
        
        if debug:
            self.PATHS = np.append(self.PATHS,
                                   [
                                       'mask',
                                       'topo_ref',
                                       'pmf_ref',
                                       'spectrum_fg',
                                       'recon_fg',
                                       'pmf_fg',
                                   ]
                                   )

 
        self.io_create_file(self.IDXS)

    def io_create_file(self,paths):
        """
        Helper function to create file.

        Parameters
        ----------
        paths : list
            List of strings containing the name of the data sets.

        Notes
        -----
        Currently, if the filename of the HDF5 file already exists, this function will append the existing filename with '_old' and create an empty HDF5 file with the same filename in its place.

        """
        # If directory does not exist, create it.
        if not os.path.exists(self.OUTPUT_FOLDER):
            os.mkdir(self.OUTPUT_FOLDER)

        # If file exists, rename it with old.
        if os.path.exists(self.OUTPUT_FULLPATH + self.SUFFIX + self.FORMAT):
            os.rename(self.OUTPUT_FULLPATH + self.SUFFIX + self.FORMAT, self.OUTPUT_FULLPATH + self.SUFFIX + '_old' + self.FORMAT)
            
        file = h5py.File(self.OUTPUT_FULLPATH + self.SUFFIX + self.FORMAT, 'a')
        for path in paths:
            path = str(path)
            # check if groups have been created
            # if not created, create empty groups
            if not (path in file):
                file.create_group(path,track_order=True)
        
        file.close()

    def write_all(self, idx, *args):
        for arg in args:
            for attr in self.PATHS:
                if hasattr(arg, attr):
                    self.populate(idx, attr, getattr(arg, attr))

            for attr in self.ATTRS:
                if hasattr(arg, attr):
                    self.write_attr(idx, attr, getattr(arg, attr))

    def write_attr(self,idx,key,value):
        """
        Method to write attributes to HDF5 file.
        """
        file = h5py.File(self.OUTPUT_FULLPATH + self.SUFFIX + self.FORMAT, 'r+')

        try:
            file[str(idx)].attrs.create(str(key), value)
        except:
            file[str(idx)].attrs.create(str(key), repr(value),dtype='<S' + str(len(repr(value))))

        file.close()

    def write_all_attrs(self, obj):
        file = h5py.File(self.OUTPUT_FULLPATH + self.SUFFIX + self.FORMAT, 'r+')
        for key, value in vars(obj).items():
            try:
                file.attrs.create(key,value)
            except:
                file.attrs.create(key,repr(value),dtype='<S' + str(len(repr(value))))
        file.close()

    def populate(self,idx,name,data,options=None):
        """
        Helper function to write data into HDF5 dataset.

        Parameters
        ----------
        name : str
            The time and additional suffix label for the dataset
        path : str
            Path of the dataset, e.g. `rhoY`.
        data : ndarray
            The output data to write to the dataset
        options : list
            `default == None`. Additional options to write to dataset, currently unused.

        """
        # name is the simulation time of the output array
        # path is the array type, e.g. U,V,H, and data is it's data.
        file = h5py.File(self.OUTPUT_FULLPATH + self.SUFFIX + self.FORMAT, 'r+')

        path = str(idx) + '/' + str(name)
        if not (path in file):
            file.create_dataset(path, data=data, chunks=True, compression='gzip', compression_opts=4)
        else:
            file[path][...] = data

        file.close()


class reader(object):

    def __init__(self, fn):
        self.fn = fn

        self.names = {
                        'lat' : 'lat',
                        'lon' : 'lon',
                        'recon' : 'data',
                        'ampls' : 'spec',
                        'pmf_sg' : 'pmf',
                    }

    def get_params(self, params):
        file = h5py.File(self.fn)

        for key in file.attrs.keys():
            setattr(params, key, file.attrs[key])

        file.close()

    def read_data(self, idx, cell):
        file = h5py.File(self.fn)

        idx = str(idx)
        for key, value in self.names.items():
            setattr(cell, value, file[idx][key][:])

        file.close()


