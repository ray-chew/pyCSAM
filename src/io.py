import netCDF4 as nc
import numpy as np
import h5py
import os

class ncdata(object):

    def __init__(self):
        None

    def read_dat(self, fn, obj):
        df = nc.Dataset(fn)

        for key, _ in vars(obj).items():
            if key in df.variables:
                setattr( obj, key, df.variables[key][:] )

        df.close()

    def read_topo(self, topo, cell, lon_vert,lat_vert):
        
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
