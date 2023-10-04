import netCDF4 as nc
import numpy as np
import h5py
import os

from src import utils

class ncdata(object):

    def __init__(self, read_merit = False):
        self.read_merit = read_merit

    def read_dat(self, fn, obj):
        df = nc.Dataset(fn)

        for key, _ in vars(obj).items():
            if key in df.variables:
                setattr( obj, key, df.variables[key][:] )

        df.close()

    def read_topo(self, topo, cell, lon_vert, lat_vert):
        
        #---- variable and the coordinates
        lon, lat, z = topo.lon, topo.lat, topo.topo
        
        #---- get number of records,nlat,nlon
        nrecords = np.shape(z)[0]; nlon = np.shape(lon)[1]; nlat = np.shape(lat)[1]
        
        #---- process each record to get the (lon,lat) for each topographic observation in 1D
        lon_res=[]; lat_res=[]; z_res=[]   # resulting lon,lat,z  
        
        for n in range(nrecords):
            # print('n = ',n)
            lon_,lat_ = np.meshgrid(lon[n][:],lat[n][:])
    #         print(lon_, lat_)
            lon_= lon_.ravel() 
            lat_ = lat_.ravel() 
            z_ = np.flipud(z[n][:]).ravel() 
    #         print(lon_.shape, lat_.shape, z_.shape)
            cond_lat = ( lat_vert.min() <= lat_ ) & ( lat_ <= lat_vert.max() )
            cond_lon = ( lon_vert.min() <= lon_ ) & ( lon_ <= lon_vert.max() )
            idx = np.nonzero((cond_lat & cond_lon))[0]

            # idx = np.nonzero((np.abs(lon_ - lon_centre)<= lon_width/2) & 
                            # (np.abs(lat_ - lat_centre)<= lat_width/2))[0]
            # print(idx)
            if len(idx)!=0:
                lon_dummy,lat_dummy,z_dummy = lon_[idx],lat_[idx],z_[idx]
                lon_res.extend(lon_dummy.tolist())
                lat_res.extend(lat_dummy.tolist())
                z_res.extend(z_dummy.tolist())
        
        lon_res = np.array(lon_res)
        lat_res = np.array(lat_res)
        z_res = np.array(z_res)
            
        del lat, lon, z
            
        #---- processing of the lat,lon,topo to get the regular 2D grid for topography
        lon_uniq, lat_uniq = np.unique(lon_res), np.unique(lat_res) # get unique values of lon,lat
        nla = len(lat_uniq); nlo = len(lon_uniq)
        
        # print("lat_res shape = ", lat_res.shape)
        # print("lon_res shape = ", lon_res.shape)
        # print("z_res shape = ", z_res.shape)
        # print("nla = ", nla)
        # print("nlo = ", nlo)

        #---- building 2D topography field
        lat_lon_topo = np.vstack((lat_res,lon_res,z_res)).T
        lat_lon_topo = lat_lon_topo[lat_lon_topo[:,0].argsort()]  # sorted according to latitude
        lon_sort_id = [lat_lon_topo[n*nlo:(n+1)*nlo,1].argsort()+nlo*n for n in range(nla)] 
        lon_sort_id = np.array(lon_sort_id).reshape(-1)
        lat_lon_topo = lat_lon_topo[lon_sort_id]  # sorted according to longitude for each len(lon_u)
        topo_2D = np.reshape(lat_lon_topo[:,2],(nla,nlo))
        del lat_lon_topo, lon_sort_id

        print('Data fetched...')
        cell.lon = lon_uniq
        cell.lat = lat_uniq
        cell.topo = topo_2D


    class read_merit_topo(object):
        
        def __init__(self, cell, dir, lat_verts, lon_verts):
            self.dir = dir

            self.fn_lon = np.array([-180.0, -150.0, -120.0, -90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0])
            self.fn_lat = np.array([90.0, 60.0, 30.0, 0.0, -30.0, -60.0, -90.0])

            self.lat_verts = lat_verts
            self.lon_verts = lon_verts

            lat_min_idx = self.compute_idx(lat_verts.min(), 'min', 'lat')
            lat_max_idx = self.compute_idx(lat_verts.max(), 'max', 'lat')

            lon_min_idx = self.compute_idx(lon_verts.min(), 'min', 'lon')
            lon_max_idx = self.compute_idx(lon_verts.max(), 'max', 'lon')

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
                iint = 2
                cell.lat = np.sort(cell.lat)[::iint]
                cell.lon = np.sort(cell.lon)[::iint][:-1]

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
    def __init__(self, fn, sfx=''):
        """
        Creates HDF5 file based on filename given attribute `OUTPUT_FILENAME`.

        """

        self.FORMAT = ".h5"
        self.OUTPUT_FOLDER = "../outputs/"
        self.OUTPUT_FILENAME = fn
        self.OUTPUT_FULLPATH = self.OUTPUT_FOLDER + self.OUTPUT_FILENAME
        self.SUFFIX = sfx
                
        self.PATHS = [  'coarse_errs',
                        'fine_errs',
                        'coarse_lmbdas',
                        'fine_lmbdas',
                        'min_cls',
                        'min_fls',
                        'min_errs_cls',
                        'min_errs_fls',
                        'opt_lmbdas',
                        'opt_errs',
                        'dat_errs',
                    ]
 
        self.io_create_file(self.PATHS)

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
            os.rename(self.OUTPUT_FULLPATH + self.SUFFIX + self.FORMAT, self.FULLPATH + self.SUFFIX + '_old' + self.FORMAT)
            
        file = h5py.File(self.OUTPUT_FULLPATH + self.SUFFIX + self.FORMAT, 'a')
        for path in paths:
            # check if groups have been created
            # if not created, create empty groups
            if not (path in file):
                file.create_group(path,track_order=True)
        
        file.close()


    # def write_all(self,Sol,mpv,elem,node,th,name):
    #     """
    #     At a given time, write output from `Sol` and `mpv` to the HDF5 file.

    #     Parameters
    #     ----------
    #     Sol : :class:`management.variable.Vars`
    #         Solution data container
    #     mpv : :class:`physics.low_mach.mpv.MPV`
    #         Variables relating to the elliptic solver
    #     elem : :class:`discretization.kgrid.ElemSpaceDiscr`
    #         Cells grid
    #     node : :class:`discretization.kgrid.NodeSpaceDiscr`
    #         Nodes grid
    #     th : :class:`physics.gas_dynamics.thermodynamic.ThermodynamicInit`
    #         Thermodynamic variables of the system
    #     name: str
    #         The time and additional suffix label for the dataset, e.g. "_10.0_after_full_step", where 10.0 is the time and "after_full_step" denotes when the output was made.

    #     """
    #     print("writing hdf output..." + name)
    #     self.populate(name,'rho',Sol.rho)
    #     self.populate(name,'rhoY',Sol.rhoY)
    #     self.populate(name,'rhou',Sol.rhou)
    #     self.populate(name,'rhov',Sol.rhov)
    #     self.populate(name,'rhow',Sol.rhow)
    #     self.populate(name,'rhoX',Sol.rhoX)

    #     self.populate(name,'p2_nodes',mpv.p2_nodes)
    #     self.populate(name,'vorty', self.vorty(Sol,elem,node))


    def populate(self,name,path,data,options=None):
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
        file.create_dataset(str(path) + '/' + str(path) + '_' + str(name), data=data, chunks=True, compression='gzip', compression_opts=4, dtype=np.float32)

        file.close()
