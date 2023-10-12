import numpy as np
from src import utils

class grid(object):
    def __init__(self):
        self.clat = None
        self.clat_vertices = None
        self.clon = None
        self.clon_vertices = None
        self.links = None

    def apply_f(self, f):
        self.non_convertibles = ['non_convertibles', 'links']
        for key, value in vars(self).items():
            if key in self.non_convertibles:
                pass
            else:
                setattr(self, key, f(value))



class topo(object):
    def __init__(self):
        self.lon = None
        self.lat = None
        self.topo = None
        self.analysis = None


class topo_cell(topo):
    def __init__(self):
        super().__init__()


    def gen_mgrids(self, grad=False):
        if not grad:
            lat, lon = self.lat, self.lon
            self.lon_grid, self.lat_grid = np.meshgrid(lon, lat)
        else:
            lat, lon = self.lat, self.lon
            grad_lat, grad_lon = self.grad_lat, self.grad_lon
            self.grad_lat_lon_grid, self.grad_lat_lat_grid = np.meshgrid(lon, grad_lat)
            self.grad_lon_lon_grid, self.grad_lon_lat_grid = np.meshgrid(grad_lon, lat)

    
    def __get_lat_lon_points(self, grad=False):
        if not grad:
            lat_grid, lon_grid = self.lat_grid, self.lon_grid
        else:
            lat_grid, lon_grid = self.grad_lat_grid, self.grad_lon_grid

        lat_grid_tmp = np.expand_dims(np.copy(lat_grid),-1)
        lon_grid_tmp = np.expand_dims(np.copy(lon_grid),-1)

        lat_grid_tmp = utils.rescale(lat_grid_tmp)
        lon_grid_tmp = utils.rescale(lon_grid_tmp)

        return np.stack((lon_grid_tmp, lat_grid_tmp), axis=2).reshape(-1,2)


    def __get_mask(self, triangle):
        lat_lon_points = self.__get_lat_lon_points()
        init_poly = triangle.vec_get_mask

        self.mask = np.array([init_poly(elem) for elem in lat_lon_points]).reshape(self.topo.shape).astype('bool_')


    def get_masked(self, triangle = None, mask = None):

        if (triangle is not None) and (mask is None):
            self.__get_mask(triangle)
        elif (mask is not None):
            self.mask = mask

        self.lon_m = self.lon_grid[self.mask]
        self.lat_m = self.lat_grid[self.mask]
        self.topo_m = self.topo[self.mask]


    def get_grad_topo(self, triangle):
        lat, lon = self.lat, self.lon
        self.grad_lat = lat[:-1] + 0.5 * (lat[1:] - lat[:-1])
        self.grad_lon = lon[:-1] + 0.5 * (lon[1:] - lon[:-1])

        self.gen_mgrids(grad=True)

        dlat = np.diff(self.lat).reshape(1,-1)
        dlon = np.diff(self.lon).reshape(-1,1)

        grad_lon_topo = (self.topo[1:,:] - self.topo[:-1,:]) / dlon
        grad_lat_topo = (self.topo[:,1:] - self.topo[:,:-1]) / dlat

        lat_lon_points = self.__get_lat_lon_points(grad=True)
        init_poly = triangle.vec_get_mask

        self.grad_mask = np.array([init_poly(elem) for elem in lat_lon_points]).reshape(self.topo.shape).astype('bool_')

        grad_lon_topo = grad_lon_topo[self.grad_mask]
        grad_lat_topo = grad_lat_topo[self.grad_mask]


        self.grad_lon_m = self.grad_lon_grid[self.grad_mask]
        self.grad_lat_m = self.grad_lat_grid[self.grad_mask]
        self.grad_topo_m = np.vstack([grad_lon_topo, grad_lat_topo])


class analysis(object):
    def __init__(self):
        self.wlat = None
        self.wlon = None
        self.ampls = None

        # only works with explicitly setting the (k,l)-values
        self.kks = None
        self.lls = None

        self.recon = None

    def get_attrs(self, fobj, freqs):
        self.wlat = np.copy(fobj.wlat)
        self.wlon = np.copy(fobj.wlon)
        self.ampls = np.copy(freqs)

        # only works with explicitly setting the (k,l)-values
        # if hasattr(fobj, 'k_idx'):
        #     self.kks = fobj.k_idx / (fobj.Ni)# / np.sqrt(2.0))
        # else:
        #     self.kks = fobj.m_i / (fobj.Ni)# / np.sqrt(2.0))
        # if hasattr(fobj, 'l_idx'):
        #     self.lls = fobj.l_idx / (fobj.Nj)# / np.sqrt(2.0))
        # else:
        #     self.lls = fobj.m_j / (fobj.Nj)# / np.sqrt(2.0))

        #     pts = []
        #     cnt = 0
        #     for ll in self.lls:
        #         for kk in self.kks:
        #             if kk == 0 and ll <= 0:
        #                 continue
        #             else:
        #                 pts.append([kk,ll])

        #             if int(kk) == 0 and int(ll) == 0:
        #                 idx = cnt
                    
        #             cnt += 1

        #     pts = np.array(pts)
        #     self.kks = pts[:,0]
        #     self.lls = pts[:,1]

        #     self.ampls = np.delete(self.ampls, idx)

        self.kks = fobj.m_i / (fobj.Ni)
        self.lls = fobj.m_j / (fobj.Nj)

        self.kks, self.lls = np.meshgrid(self.kks, self.lls)


        # self.kks = self.kks / self.kks.size
        # self.lls = self.lls / self.lls.size

#         self.clat = ma.getdata(df.variables['clat'][:])
# clat_vertices = ma.getdata(df.variables['clat_vertices'][:])
# clon = ma.getdata(df.variables['clon'][:])
# clon_vertices = ma.getdata(df.variables['clon_vertices'][:])

    def grid_kk_ll(self, fobj, dat):

        m_i = fobj.m_i
        m_j = fobj.m_j

        freq_grid = np.zeros((len(m_i), len(m_j)))

        cnt = 0
        for l_idx, ll in enumerate(m_j):
            for k_idx, kk in enumerate(m_i):
                print(kk, ll, k_idx, l_idx, cnt)
                if kk == 0 and ll <= 0:
                    freq_grid[l_idx, k_idx] = 0.0
                else:
                    freq_grid[l_idx, k_idx] = dat[cnt]
                    cnt += 1


        return freq_grid
    

class obj(object):
    pass