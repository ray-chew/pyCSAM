"""
This module defines the data objects used in the program.
"""

import numpy as np
from src import utils, io

class grid(object):
    """
    Grid class
    """
    def __init__(self):
        """
        Contains the ``(lat,lon)`` of each triangular grid cell with the corresponding vertices ``(lat_1, lat_2, lat_3)``, ``(lon_1, lon_2, lon_3)``.

        ``link`` is a lookup table linking the grid cell to the corresponding topography file.
        """
        self.clat = None
        self.clat_vertices = None
        self.clon = None
        self.clon_vertices = None
        self.links = None

    def apply_f(self, f):
        """
        Applies a function to all class attributes, except those listed in ``non_convertibles``

        Parameters
        ----------
        f : ``function``
            arbitrary function to be applied to class attributes, e.g. a radians-degrees converter.
        """
        self.non_convertibles = ['non_convertibles', 'links']
        for key, value in vars(self).items():
            if key in self.non_convertibles:
                pass
            else:
                setattr(self, key, f(value))



class topo(object):
    """
    Topography class with its corresponding lat-lon values
    """
    def __init__(self):
        self.lon = None
        self.lat = None
        self.topo = None
        self.analysis = None


class topo_cell(topo):
    """
    Inherits and initialises an instance of :class:`src.var.topo`, to be used for storing data associated to a grid cell
    """
    def __init__(self):
        super().__init__()


    def gen_mgrids(self, grad=False):
        """
        Generates a meshgrid based on the lat-lon values

        Parameters
        ----------
        grad : bool, optional
            deprecated by 0.90.0, by default False
        """
        if not grad:
            lat, lon = self.lat, self.lon
            self.lon_grid, self.lat_grid = np.meshgrid(lon, lat)
        else:
            lat, lon = self.lat, self.lon
            grad_lat, grad_lon = self.grad_lat, self.grad_lon
            self.grad_lat_lon_grid, self.grad_lat_lat_grid = np.meshgrid(lon, grad_lat)
            self.grad_lon_lon_grid, self.grad_lon_lat_grid = np.meshgrid(grad_lon, lat)

    
    def __get_lat_lon_points(self, grad=False):
        """
        Private method to get the (lat,lon) coordinate for each topographic data point
        """
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
        """
        Private method to generate the mask based on which data points are inside the triangle grid cell.

        Parameters
        ----------
        triangle : :class:`src.utils.gen_triangle`
            instance of the generate-triangle class
        """
        lat_lon_points = self.__get_lat_lon_points()
        init_poly = triangle.vec_get_mask

        self.mask = np.array([init_poly(elem) for elem in lat_lon_points]).reshape(self.lat.size, self.lon.size).astype('bool_')


    def get_masked(self, triangle = None, mask = None):
        """Gets the masked attributes

        Parameters
        ----------
        triangle : :class:`src.utils.gen_triangle`
            instance of the generate-triangle class, by default None
        mask : array-like, optional
            2D array of the mask, by default None
        """

        if (triangle is not None) and (mask is None):
            self.__get_mask(triangle)
        elif (mask is not None):
            self.mask = mask

        self.lon_m = self.lon_grid[self.mask]
        self.lat_m = self.lat_grid[self.mask]
        self.topo_m = self.topo[self.mask]
        
        self.topo_m -= self.topo_m.mean()


    def get_grad_topo(self, triangle):
        """
        Computes the gradient of the topography

        .. deprecated:: 0.90.0

        """
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
    """
    Analysis object, contains all the attributes required to compute the idealised pseudo-momentum fluxes

    """
    def __init__(self):
        """
        Initialises empty attributes
        """
        self.wlat = None
        self.wlon = None
        self.ampls = None

        # only works with explicitly setting the (k,l)-values
        self.kks = None
        self.lls = None

        self.recon = None

    def get_attrs(self, fobj, freqs):
        """Copies required attributes given the arguments

        Parameters
        ----------
        fobj : :class:`src.fourier.f_trans`
            instance of the Fourier transformer
        freqs : array-like
            2D (abs. valued real) spectrum 
        """
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
        """
        .. deprecated:: 0.90.0

        """
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
    """Helper object to generate class instances on the fly
    """
    def __init__(self):
        pass

    def print(self):
        for var in vars(self):
            print(var, getattr(self, var))


class params(obj):
    """User parameter class

    Defines required and optional parameters to run a simulation
    """
    def __init__(self):
        """
        Defines the required parameters for a simulation run
        """
        # Define filenames
        self.path = '/home/ray/git-projects/spec_appx/data/'
        self.fn_grid = self.path + 'icon_compact.nc'
        self.fn_topo = self.path + 'topo_compact.nc'

        self.output_fn = None

        self.enable_merit = True
        self.merit_cg = 10
        self.merit_path = '/home/ray/Documents/orog_data/MERIT/'

        # Domain size
        self.lat_extent = None
        self.lon_extent = None

        self.run_full_land_model = True

        # Compulsory Delaunay parameters
        self.delaunay_xnp = None
        self.delaunay_ynp = None
        self.rect_set = None
        self.lxkm, self.lykm = None, None

        # Set the Fourier parameters and object.
        self.nhi = 24
        self.nhj = 48
        self.n_modes = 100

        # Set artificial wind
        self.U, self.V = 10.0, 0.1

        # Set Spec Appx parameters
        self.rect = True
        self.dfft_first_guess = False
        self.refine = False
        self.no_corrections = True
        self.cg_spsp = False # coarse grain the spectral space?
        self.rect = False if self.cg_spsp else True 

        self.fa_iter_solve = True
        self.sa_iter_solve = True

        # Penalty terms
        self.lmbda_fa = 1e-2 # first guess
        self.lmbda_sa = 1e-1 # second step

        # Tapering parameters
        self.taper_ref       = False
        self.taper_fa        = False
        self.taper_sa        = False
        self.taper_art_it    = 50
        self.padding = 0 # must be less than 60

        # Flags
        self.get_delaunay_triangulation = False
        self.debug = False
        self.debug_writer = True
        self.verbose = False
        self.plot = False


    def self_test(self):
        """
        Checker method if user-defined parameters contains sensible compulsory parameters. Calls :func:`src.var.params.check_init` and :func:`src.var.params.check_delaunay`.

        Returns
        -------
        bool
            True if test passed, False otherwise
        """
        if self.output_fn is None:
            self.output_fn = io.fn_gen(self)

        self.check_init()

        if self.get_delaunay_triangulation:
            self.check_delaunay()

        return True

    def check_init(self):
        """Checks if all required parameters are defined.
        """
        compulsory_params = [
                             'lat_extent',
                             'lon_extent'
                             ]
        
        offenders = self.checker(self, compulsory_params)
        assert len(offenders) == 0, "Compulsory run parameter(s) undefined: %s" %offenders

    def check_delaunay(self):
        """
        If run uses a Delaunay triangulation, this method checks if all required parameters are defined.
        """
        compulsory_params = [
                             'delaunay_xnp',
                             'delaunay_ynp',
                             'rect_set',
                             'lxkm',
                             'lykm'
                             ]

        offenders = self.checker(self, compulsory_params)
        assert len(offenders) == 0, "Compulsory Delaunay run parameter(s) undefined: %s" %offenders


    @staticmethod
    def checker(arg, compulsory_params):
        """Auxiliary function that checks if ``arg`` is in ``compulsory_params``

        """
        offenders = []
        for key, value in vars(arg).items():
            if key in compulsory_params:
                if value is None:
                    offenders.append(key)
        return offenders