import numpy as np

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


    def gen_mgrids(self):
        self.lon_grid, self.lat_grid = np.meshgrid(self.lon, self.lat)


    def __get_mask(self, triangle):
        self.mask = triangle.vec_get_mask(self.lon_grid.ravel(), self.lat_grid.ravel())
        self.mask = self.mask.reshape(self.topo.shape)


    def get_masked(self, triangle = None, mask = None):

        if (triangle is not None) and (mask is None):
            self.__get_mask(triangle)
        elif (mask is not None):
            self.mask = mask

        # self.lon_m = np.ma.masked_array(self.lon_grid, mask=self.mask)
        # self.lat_m = np.ma.masked_array(self.lat_grid, mask=self.mask)
        # self.topo_m = np.ma.masked_array(self.topo, mask=self.mask)

        self.lon_m = self.lon_grid[self.mask]
        self.lat_m = self.lat_grid[self.mask]
        self.topo_m = self.topo[self.mask]


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
        self.wlat = fobj.wlat
        self.wlon = fobj.wlon
        self.ampls = freqs

        # only works with explicitly setting the (k,l)-values
        self.kks = fobj.k_idx / (fobj.Ni / 2.0)
        self.lls = fobj.l_idx / (fobj.Nj / 2.0)

        # self.kks = self.kks / self.kks.size
        # self.lls = self.lls / self.lls.size

#         self.clat = ma.getdata(df.variables['clat'][:])
# clat_vertices = ma.getdata(df.variables['clat_vertices'][:])
# clon = ma.getdata(df.variables['clon'][:])
# clon_vertices = ma.getdata(df.variables['clon_vertices'][:])