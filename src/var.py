import numpy as np

class grid(object):
    def __init__(self):
        self.clat = None
        self.clat_vertices = None
        self.clon = None
        self.clon_vertices = None


    def apply_f(self, f):
        for key, value in vars(self).items():
            setattr(self, key, f(value))



class topo(object):
    def __init__(self):
        self.lon = None
        self.lat = None
        self.topo = None


class topo_cell(topo):
    def __init__(self):
        super().__init__()


    def gen_mgrids(self):
        self.lon_grid, self.lat_grid = np.meshgrid(self.lon, self.lat)


    def __get_mask(self, triangle):
        self.mask = triangle.vec_get_mask(self.lon_grid.ravel(), self.lat_grid.ravel())
        self.mask = self.mask.reshape(self.topo.shape)


    def get_masked(self, triangle):

        self.__get_mask(triangle)

        # self.lon_m = np.ma.masked_array(self.lon_grid, mask=self.mask)
        # self.lat_m = np.ma.masked_array(self.lat_grid, mask=self.mask)
        # self.topo_m = np.ma.masked_array(self.topo, mask=self.mask)

        self.lon_m = self.lon_grid[self.mask]
        self.lat_m = self.lat_grid[self.mask]
        self.topo_m = self.topo[self.mask]




#         self.clat = ma.getdata(df.variables['clat'][:])
# clat_vertices = ma.getdata(df.variables['clat_vertices'][:])
# clon = ma.getdata(df.variables['clon'][:])
# clon_vertices = ma.getdata(df.variables['clon_vertices'][:])