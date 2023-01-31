import netCDF4 as nc
import numpy as np

class writer(object):

    def __init__(self):
        None

    def read_dat(self, fn, obj):
        df = nc.Dataset(fn)

        for key, _ in vars(obj).items():
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
            