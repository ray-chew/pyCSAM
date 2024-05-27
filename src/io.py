"""
Input/Output routines
"""

import netCDF4 as nc
import numpy as np
import h5py
import os
from datetime import datetime

from ..src import utils


class ncdata(object):
    """Helper class to read NetCDF4 topographic data"""

    def __init__(self, read_merit=False, padding=0, padding_tol=50):
        """

        Parameters
        ----------
        read_merit : bool, optional
            toggles between the `MERIT DEM <https://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_DEM/>`_ and `USGS GMTED 2010 <https://www.usgs.gov/coastal-changes-and-impacts/gmted2010>`_ data files. By default False, i.e., read USGS GMTED 2010 data files.
        padding : int, optional
            number of data points to pad the loaded topography file, by default 0
        padding_tol : int, optional
            padding tolerance is added no matter the user-defined ``padding``, by default 50
        """
        self.read_merit = read_merit
        self.padding = padding_tol + padding
        self.is_open = False

    def read_dat(self, fn, obj):
        """Reads data by attributes defined in the ``obj`` class.

        Parameters
        ----------
        fn : str
            filename
        obj : :class:`src.var.grid` or :class:`src.var.topo` or :class:`src.var.topo_cell`
            any data object in :mod:`src.var` accepting topography attributes
        """
        df = nc.Dataset(fn, "r")

        for key, _ in vars(obj).items():
            if key in df.variables:
                setattr(obj, key, df.variables[key][:])

        df.close()

    def open(self, fn):
        self.df = nc.Dataset(fn, "r")
        self.is_open = True

    def close(self):
        if self.is_open and hasattr(self, "df"):
            self.df.close()

    def __get_truths(self, arr, vert_pts, d_pts):
        """Assembles Boolean array selecting for data points within a given lat-lon range, including padded boundary."""
        return (arr >= (vert_pts.min() - self.padding * d_pts)) & (
            arr <= vert_pts.max() + self.padding * d_pts
        )

    def read_topo(self, topo, cell, lon_vert, lat_vert):
        """Reads USGS GMTED 2010 dataset

        Parameters
        ----------
        topo : :class:`src.var.topo` or :class:`src.var.topo_cell`
            instance of a topography class containing the full regional or global topography loaded via :func:`src.io.read_dat`.
        cell : :class:`src.var.topo_cell`
            instance of a cell object
        lon_vert : list
            extent of the longitudinal coordinates encompassing the region to be loaded
        lat_vert : list
            extent of the latitudinal coordinates encompassing the region to be loaded

        .. note:: Loading the global topography in the ``topo`` argument may not be memory efficient. The notebook ``nc_compactifier.ipynb`` contains a script to extract a region of interest from the global GMTED 2010 dataset.
        """
        lon, lat, z = topo.lon, topo.lat, topo.topo

        nrecords = np.shape(z)[0]

        bool_arr = np.zeros_like(z).astype(bool)
        lat_arr = np.zeros_like(z)
        lon_arr = np.zeros_like(z)

        z = z[:, ::-1, :]

        for n in range(nrecords):
            lat_n = lat[n]
            lon_n = lon[n]

            dlat, dlon = np.diff(lat_n).mean(), np.diff(lon_n).mean()

            lon_nm, lat_nm = np.meshgrid(lon_n, lat_n)

            bool_arr[n] = self.__get_truths(lon_nm, lon_vert, dlon) & self.__get_truths(
                lat_nm, lat_vert, dlat
            )

            lat_arr[n] = lat_nm
            lon_arr[n] = lon_nm

        lon_res = lon_arr[bool_arr]
        lat_res = lat_arr[bool_arr]
        z_res = z[bool_arr].data

        # ---- processing of the lat,lon,topo to get the regular 2D grid for topography
        lon_uniq, lat_uniq = np.unique(lon_res), np.unique(
            lat_res
        )  # get unique values of lon,lat
        nla = len(lat_uniq)
        nlo = len(lon_uniq)

        lat_res_sort_idx = np.argsort(lat_res)
        lon_res_sort_idx = np.argsort(
            lon_res[lat_res_sort_idx].reshape(nla, nlo), axis=1
        )
        z_res = z_res[lat_res_sort_idx]
        z_res = np.take_along_axis(z_res.reshape(nla, nlo), lon_res_sort_idx, axis=1)
        topo_2D = z_res.reshape(nla, nlo)

        print("Data fetched...")
        cell.lon = lon_uniq
        cell.lat = lat_uniq
        cell.topo = topo_2D

    class read_merit_topo(object):
        """Subclass to read MERIT topographic data"""

        def __init__(self, cell, params, verbose=False, is_parallel=False):
            """Populates ``cell`` object instance with arguments from ``params``

            Parameters
            ----------
            cell : :class:`src.var.topo` or :class:`src.var.topo_cell`
                instance of an object with topograhy attribute
            params : :class:`src.var.params`
                user-defined run parameters
            verbose : bool, optional
                prints loading progression, by default False
            """
            self.dir = params.path_merit
            self.verbose = verbose
            self.opened_dfs = []

            self.fn_lon = np.array(
                [
                    -180.0,
                    -150.0,
                    -120.0,
                    -90.0,
                    -60.0,
                    -30.0,
                    0.0,
                    30.0,
                    60.0,
                    90.0,
                    120.0,
                    150.0,
                    180.0
                ]
            )
            self.fn_lat = np.array([90.0, 60.0, 30.0, 0.0, -30.0, -60.0, -90.0])

            self.lat_verts = np.array(params.lat_extent)
            self.lon_verts = np.array(params.lon_extent)

            self.merit_cg = params.merit_cg
            self.split_EW = False

            if not is_parallel:
                self.get_topo(cell)
            
            self.is_parallel = is_parallel

        def get_topo(self, cell):

            # if lat_verts 

            if ( (self.lon_verts.max() - self.lon_verts.min()) > 180.0 ):
                self.split_EW = True

            if self.split_EW:
                min_lon = max(np.where(self.lon_verts < 0.0, self.lon_verts + 360.0, self.lon_verts)) - 360.0
                max_lon = min(np.where(self.lon_verts < 0.0, self.lon_verts + 360.0, self.lon_verts))
            else:
                min_lon = self.lon_verts.min()
                max_lon = self.lon_verts.max()

            lat_min_idx = self.__compute_idx(self.lat_verts.min(), "min", "lat")
            lat_max_idx = self.__compute_idx(self.lat_verts.max(), "max", "lat")

            lon_min_idx = self.__compute_idx(min_lon, "min", "lon")
            lon_max_idx = self.__compute_idx(max_lon, "max", "lon")

            if ( (self.lon_verts.max() - self.lon_verts.min()) > 180.0 ):
                lon_idx_rng = list(range(lon_max_idx, len(self.fn_lon) - 1 )) + list(range(0,lon_min_idx + 1))

            else:
                if lon_min_idx == lon_max_idx:
                    lon_max_idx += 1
                lon_idx_rng = list(range(lon_min_idx, lon_max_idx))

            lat_idx_rng = list(range(lat_max_idx, lat_min_idx))

            fns, dirs, lon_cnt, lat_cnt = self.__get_fns(
                lat_idx_rng, lon_idx_rng
            )

            self.__load_topo(cell, fns, dirs, lon_cnt, lat_cnt, lat_idx_rng, lon_idx_rng)

        def __compute_idx(self, vert, typ, direction):
            """Given a point ``vert``, look up which MERIT NetCDF file contains this point."""
            if direction == "lon":
                fn_int = self.fn_lon
            else:
                fn_int = self.fn_lat

            where_idx = np.argmin(np.abs(fn_int - vert))

            if self.verbose:
                print(fn_int, where_idx)

            if typ == "min":
                if ((vert - fn_int[where_idx]) < 0.0):
                    if direction == "lon":
                        if not self.split_EW:
                            where_idx -= 1
                    else:
                        where_idx += 1
            elif typ == "max":
                if ((vert - fn_int[where_idx]) > 0.0):
                    if direction == "lon":
                        if not self.split_EW:
                            where_idx += 1
                    else:
                        where_idx -= 1

                if where_idx == (len(fn_int) - 1):
                    where_idx -= 1

            where_idx = int(where_idx)

            if self.verbose:
                print("where_idx, vert, fn_int[where_idx] for typ:")
                print(where_idx, vert, fn_int[where_idx], typ)
                print("")

            return where_idx

        def __get_fns(self, lat_idx_rng, lon_idx_rng):
            """Construct the full filenames required for the loading of the topographic data from the indices identified in :func:`src.io.ncdata.read_merit_topo.__compute_idx`"""
            fns = []
            dirs = []

            for lat_cnt, lat_idx in enumerate(lat_idx_rng):
                l_lat_bound, r_lat_bound = (
                    self.fn_lat[lat_idx],
                    self.fn_lat[lat_idx + 1],
                )
                l_lat_tag, r_lat_tag = self.__get_NSEW(
                    l_lat_bound, "lat"
                ), self.__get_NSEW(r_lat_bound, "lat")

                if ((l_lat_tag == "S" and r_lat_tag == "S") and (l_lat_bound == -60 and r_lat_bound == -90)):
                    merit_or_rema = "REMA_BKG"
                    self.rema = True
                    self.dir = self.dir.replace("MERIT", "REMA")
                else:
                    merit_or_rema = "MERIT"
                    self.rema = False
                    self.dir = self.dir.replace("REMA", "MERIT")

                for lon_cnt, lon_idx in enumerate(lon_idx_rng):
                    l_lon_bound, r_lon_bound = (
                        self.fn_lon[lon_idx],
                        self.fn_lon[lon_idx + 1],
                    )
                    l_lon_tag, r_lon_tag = self.__get_NSEW(
                        l_lon_bound, "lon"
                    ), self.__get_NSEW(r_lon_bound, "lon")

                    name = "%s_%s%.2d-%s%.2d_%s%.3d-%s%.3d.nc4" % (
                        merit_or_rema,
                        l_lat_tag,
                        np.abs(l_lat_bound),
                        r_lat_tag,
                        np.abs(r_lat_bound),
                        l_lon_tag,
                        np.abs(l_lon_bound),
                        r_lon_tag,
                        np.abs(r_lon_bound),
                    )

                    fns.append(name)
                    dirs.append(self.dir)

            return fns, dirs, lon_cnt, lat_cnt

        def __load_topo(self, cell, fns, dirs, lon_cnt, lat_cnt,  lat_idx_rng, lon_idx_rng, init=True, populate=True):
            """
            This method assembles a contiguous array in ``cell.topo`` containing the regional topography to be loaded.

            However, this full regional array is assembled from an array of block arrays. Each block array is loaded from a separated MERIT data file and varies in shape that is not known beforehand.

            Therefore, the ``get_topo`` method is run recursively:
                1. The first run determines the shape of each constituting block array and subsequently the shape of the full regional array. An empty array in initialised.
                2. The second run populates the empty array with the information of the block arrays obtained in the first run.
            """
            if (cell.topo is None) and (init):
                self.__load_topo(cell, fns, dirs, lon_cnt, lat_cnt, lat_idx_rng, lon_idx_rng, init=False, populate=False)

            if not populate:
                n_col = 0
                n_row = 0
                nc_lon = 0
                nc_lat = 0
            else:
                n_col = 0
                n_row = 0
                lon_sz_old = 0
                lat_sz_old = 0
                cell.lat = []
                cell.lon = []

            ### Handles the case where a cell spans four topographic datasets
            cnt_lat = 0
            cnt_lon = 0
            lat_low_old = np.ones((len(fns))) * np.inf
            lat_high_old = np.ones((len(fns))) * np.inf
            lon_low_old = np.ones((len(fns))) * np.inf
            lon_high_old = np.ones((len(fns))) * np.inf
            lat_nc_change, lon_nc_change = False, False

            for cnt, fn in enumerate(fns):
                # try:
                #     test.isopen()
                # except:
                test = nc.Dataset(dirs[cnt] + fn, "r")
                self.opened_dfs.append(test)

                lat = test["lat"]
                lat_min_idx = np.argmin(np.abs((lat - np.sign(lat) * 1e-4) - self.lat_verts.min()))
                lat_max_idx = np.argmin(np.abs((lat + np.sign(lat) * 1e-4) - self.lat_verts.max()))

                lat_high = np.max((lat_min_idx, lat_max_idx))
                lat_low = np.min((lat_min_idx, lat_max_idx))

                # lon = test["lon"]
                # lon_min_idx = np.argmin(np.abs((lon - np.sign(lon) * 1e-4) - (self.lon_verts.min())))
                # lon_max_idx = np.argmin(np.abs((lon + np.sign(lon) * 1e-4) - (self.lon_verts.max())))

                # lon_high = np.max((lon_min_idx, lon_max_idx))
                # lon_low = np.min((lon_min_idx, lon_max_idx))

                ### Only add lat and lon elements if there are changes to the low and high indices identified:
                # if (lon_low not in lon_low_old) and (lon_high not in lon_high_old):
                #     lon_nc_change = True

                # if (lat_low not in lat_low_old) and (lat_high not in lat_high_old):
                #     lat_nc_change = True

                ############################################
                lat = test["lat"]
                lon = test["lon"]

                l_lat_bound, r_lat_bound = (
                    self.fn_lat[lat_idx_rng[n_row]],
                    self.fn_lat[lat_idx_rng[n_row] + 1],
                )

                l_lon_bound, r_lon_bound = (
                    self.fn_lon[lon_idx_rng[n_col]],
                    self.fn_lon[lon_idx_rng[n_col] + 1],
                )

                lon_rng = r_lon_bound - l_lon_bound

                lon_in_file = self.lon_verts[( (self.lon_verts - l_lon_bound) > 0 ) & ( (self.lon_verts - l_lon_bound) <= lon_rng )]

                if len(lon_in_file) == 0:
                    lon_high = np.argmin(np.abs(lon - r_lon_bound))
                    lon_low = np.argmin(np.abs(lon - l_lon_bound))

                else:
                    if not self.split_EW:
                        if lon_in_file.max() == self.lon_verts.max():
                            lon_high = np.argmin(np.abs(lon - lon_in_file.max()))
                        else: 
                            lon_high = np.argmin(np.abs(lon - r_lon_bound))

                        if lon_in_file.min() == self.lon_verts.min():
                            lon_low = np.argmin(np.abs(lon - lon_in_file.min()))
                        else:
                            lon_low = np.argmin(np.abs(lon - l_lon_bound))

                    else:
                        if lon_in_file.max() == min(np.where(self.lon_verts < 0.0, self.lon_verts + 360.0, self.lon_verts)):
                            lon_high = np.argmin(np.abs(lon - r_lon_bound))
                            lon_low = np.argmin(np.abs(lon - lon_in_file.min()))
                        else:
                            lon_high = np.argmin(np.abs(lon - r_lon_bound))
                        
                        if lon_in_file.min() == (max(self.lon_verts[self.lon_verts < 0.0] + 360.0) - 360.0):
                            lon_high = np.argmin(np.abs(lon - lon_in_file.max()))
                            lon_low = np.argmin(np.abs(lon - l_lon_bound))
                        else:
                            lon_low = np.argmin(np.abs(lon - l_lon_bound))
                    # if r_lon_bound > lon_in_file.max():
                    #     lon_high = np.argmin(np.abs(lon - lon_in_file.max()))

                    # if lon_in_file.min() > l_lon_bound:
                    #     lon_low = np.argmin(np.abs(lon - lon_in_file.min()))


                lon_low_old[cnt] = lon_low
                lon_high_old[cnt] = lon_high
                lat_low_old[cnt] = lat_low
                lat_high_old[cnt] = lat_high

                if not populate:
                    if n_row == 0:

                    # if (cnt_lon < (lon_cnt + 1)) and lon_nc_change:
                        nc_lon += lon_high - lon_low
                        cnt_lon += 1

                    if n_col == 0:
                    # if (cnt_lat < (lat_cnt + 1)) and lat_nc_change:
                        nc_lat += lat_high - lat_low
                        cnt_lat += 1

                    n_col += 1
                    if n_col == (lon_cnt+1):
                        n_col = 0
                        n_row += 1

                else:
                    topo = test["Elevation"][lat_low:lat_high, lon_low:lon_high]
                    if n_col == 0:
                        cell.lat += lat[lat_low:lat_high].tolist()
                    if n_row == 0:
                        cell.lon += lon[lon_low:lon_high].tolist()

                    lon_sz = lon_high - lon_low
                    lat_sz = lat_high - lat_low


                    # if lon_nc_change and cnt > 0:
                    #     n_col += 1

                    # # if n_col == (lon_cnt + 1):
                    # #     n_col = 0
                    # if lat_nc_change and cnt > 0:
                    #     n_row += 1
                    #     lat_sz_old = np.copy(lat_sz)

                    cell.topo[
                        lat_sz_old : lat_sz_old + lat_sz,
                        lon_sz_old : lon_sz_old + lon_sz,
                    ] = topo

                    n_col += 1
                    lon_sz_old += np.copy(lon_sz)

                    if n_col == (lon_cnt+1):
                        n_col = 0
                        lon_sz_old = 0

                        n_row += 1
                        lat_sz_old = np.copy(lat_sz)         

                lon_nc_change = False
                lat_nc_change = False

                test.close()

            if not populate:
                cell.topo = np.zeros((nc_lat, nc_lon))
            else:

                if self.split_EW:
                    cell.lon = np.array(cell.lon)
                    cell.lon[cell.lon < 0.0] += 360.0

                iint = self.merit_cg

                cell.lat = utils.sliding_window_view(
                    np.sort(cell.lat), (iint,), (iint,)
                ).mean(axis=-1)
                cell.lon = utils.sliding_window_view(
                    np.sort(cell.lon), (iint,), (iint,)
                ).mean(axis=-1)

                cell.topo = utils.sliding_window_view(
                    cell.topo, (iint, iint), (iint, iint)
                ).mean(axis=(-1, -2))[::-1, :]


        def close_all(self):
            for df in self.opened_dfs:
                df.close()


        @staticmethod
        def __get_NSEW(vert, typ):
            """Method to determine `NSEW` in MERIT filename"""
            if typ == "lat":
                if vert >= 0.0:
                    dir_tag = "N"
                else:
                    dir_tag = "S"
            if typ == "lon":
                if vert >= 0.0:
                    dir_tag = "E"
                else:
                    dir_tag = "W"

            return dir_tag


class writer(object):
    """
    HDF5 writer class

    Contains methods to create HDF5 file, create data sets and populate them with output variables.

    .. note:: This class was taken from an I/O routine originally written for the numerical flow solver used in `Chew et al. (2022) <https://journals.ametsoc.org/view/journals/mwre/150/9/MWR-D-21-0175.1.xml>`_ and `Chew et al. (2023) <https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/an-unstable-mode-of-the-stratified-atmosphere-under-the-nontraditional-coriolis-acceleration/FFC0AE491BE3425CE829610BCF7A1933>`_.
    """

    def __init__(self, fn, idxs, sfx="", debug=False):
        """
        Creates an empty HDF5 file with filename ``fn`` and a group for each index in ``idxs``

        Parameters
        ----------
        fn : str
            filename
        idxs : list
            list of cell indices
        sfx : str, optional
            suffixes to the filename, by default ''
        debug : bool, optional
            debug flag, by default False
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
            "tri_lat_verts",
            "tri_lon_verts",
            "tri_clats",
            "tri_clons",
            "points",
            "simplices",
            # vars from the 'cell' object
            "lon",
            "lat",
            "lon_grid",
            "lat_grid",
            # vars from the 'analysis' object
            "ampls",
            "kks",
            "lls",
            "recon",
        ]

        self.ATTRS = [
            # vars from the 'analysis' object
            "wlat",
            "wlon",
        ]

        if debug:
            self.PATHS = np.append(
                self.PATHS,
                [
                    "mask",
                    "topo_ref",
                    "pmf_ref",
                    "spectrum_ref",
                    "spectrum_fg",
                    "recon_fg",
                    "pmf_fg",
                ],
            )

        self.io_create_file(self.IDXS)

    def io_create_file(self, paths):
        """
        Helper function to create file.

        Parameters
        ----------
        paths : list
            List of strings containing the name of the groups.

        Notes
        -----
        Currently, if the filename of the HDF5 file already exists, this function will append the existing filename with '_old' and create an empty HDF5 file with the same filename in its place.

        """
        # If directory does not exist, create it.
        if not os.path.exists(self.OUTPUT_FOLDER):
            os.mkdir(self.OUTPUT_FOLDER)

        # If file exists, rename it with old.
        if os.path.exists(self.OUTPUT_FULLPATH + self.SUFFIX + self.FORMAT):
            os.rename(
                self.OUTPUT_FULLPATH + self.SUFFIX + self.FORMAT,
                self.OUTPUT_FULLPATH + self.SUFFIX + "_old" + self.FORMAT,
            )

        file = h5py.File(self.OUTPUT_FULLPATH + self.SUFFIX + self.FORMAT, "a")
        for path in paths:
            path = str(path)
            # check if groups have been created
            # if not created, create empty groups
            if not (path in file):
                file.create_group(path, track_order=True)

        file.close()

    def write_all(self, idx, *args):
        """Write all attributes and datasets of a given class instance to the group ``idx``.

        Parameters
        ----------
        idx : str or int
            group name to write the attributes or datasets
        """
        for arg in args:
            for attr in self.PATHS:
                if hasattr(arg, attr):
                    self.populate(idx, attr, getattr(arg, attr))

            for attr in self.ATTRS:
                if hasattr(arg, attr):
                    self.write_attr(idx, attr, getattr(arg, attr))

    def write_attr(self, idx, key, value):
        """Write HDF5 attributes for a group

        Parameters
        ----------
        idx : str or int
            group name to write the attributes
        key : str
            attribute name
        value : any
            attribute value that is accepted by HDF5
        """
        file = h5py.File(self.OUTPUT_FULLPATH + self.SUFFIX + self.FORMAT, "r+")

        try:
            file[str(idx)].attrs.create(str(key), value)
        except:
            file[str(idx)].attrs.create(
                str(key), repr(value), dtype="<S" + str(len(repr(value)))
            )

        file.close()

    def write_all_attrs(self, obj):
        """Write all attributes a given class instance to the HDF5 file

        Parameters
        ----------
        obj : :class:`src.var.params`
            write all user-defined parameters to the HDF5 file for reproducibility of the output
        """
        file = h5py.File(self.OUTPUT_FULLPATH + self.SUFFIX + self.FORMAT, "r+")
        for key, value in vars(obj).items():
            try:
                file.attrs.create(key, value)
            except:
                file.attrs.create(key, repr(value), dtype="<S" + str(len(repr(value))))
        file.close()

    def populate(self, idx, name, data):
        """
        Helper function to write data into HDF5 dataset.

        Parameters
        ----------
        idx  : int or str
            The name of the group
        name : str
            The name of the dataset
        data : ndarray
            The output data to write to the dataset

        """
        # name is the simulation time of the output array
        # path is the array type, e.g. U,V,H, and data is it's data.
        file = h5py.File(self.OUTPUT_FULLPATH + self.SUFFIX + self.FORMAT, "r+")

        path = str(idx) + "/" + str(name)
        if not (path in file):
            file.create_dataset(
                path, data=data, chunks=True, compression="gzip", compression_opts=4
            )
        else:
            file[path][...] = data

        file.close()


class nc_writer(object):

    def __init__(self, params):

        self.fn = params.fn_output

        if self.fn[-3:] != ".nc":
            self.fn += '.nc'

        self.path = params.path_output
        self.rect_set = params.rect_set
        self.debug = params.debug_writer

        rootgrp = nc.Dataset(self.path + self.fn, "w", format="NETCDF4")
        
        for key, value in vars(params).items():

            # if params attribute is None but check passed, then the attribute is not necessary for the run; skip it
            if value is None:
                continue
            # NetCDF does not accept Boolean types; convert to int
            if type(value) is bool:
                value = int(value)
            # Else, write attribute
            setattr(rootgrp, key, value)

        _ = rootgrp.createDimension("nspec", params.n_modes)

        self.n_modes = params.n_modes
        rootgrp.close()

    def output(self, id, clat, clon, is_land, analysis=None):

        rootgrp = nc.Dataset(self.path + self.fn, "a", format="NETCDF4")

        grp = rootgrp.createGroup(str(id))

        is_land_var = grp.createVariable("is_land","i4")
        is_land_var[:] = is_land

        clat_var = grp.createVariable("clat","f8")
        clat_var[:] = clat
        clon_var = grp.createVariable("clon","f8")
        clon_var[:] = clon

        if analysis is not None:
            dk_var = grp.createVariable("dk","f8")
            dk_var[:] = analysis.dk
            dl_var = grp.createVariable("dl","f8")
            dl_var[:] = analysis.dl

            pick_idx = np.where(analysis.ampls > 0)

            H_spec_var = grp.createVariable("H_spec","f8", ("nspec",))
            H_spec_var[:] = self.__pad_zeros(analysis.ampls[pick_idx], self.n_modes)

            kks_var = grp.createVariable("kks","f8", ("nspec",))
            kks_var[:] = self.__pad_zeros(analysis.kks[pick_idx], self.n_modes)

            lls_var = grp.createVariable("lls","f8", ("nspec",))
            lls_var[:] = self.__pad_zeros(analysis.lls[pick_idx], self.n_modes)

        rootgrp.close()


    def duplicate(self, id, struct):

        rootgrp = nc.Dataset(self.path + self.fn, "a", format="NETCDF4")

        grp = rootgrp.createGroup(str(id))

        is_land_var = grp.createVariable("is_land","i4")
        is_land_var[:] = struct.is_land

        clat_var = grp.createVariable("clat","f8")
        clat_var[:] = struct.clat
        clon_var = grp.createVariable("clon","f8")
        clon_var[:] = struct.clon

        if struct.is_land:
            dk_var = grp.createVariable("dk","f8")
            dk_var[:] = struct.dk
            dl_var = grp.createVariable("dl","f8")
            dl_var[:] = struct.dl

            pick_idx = np.where(struct.ampls > 0)

            H_spec_var = grp.createVariable("H_spec","f8", ("nspec",))
            H_spec_var[:] = self.__pad_zeros(struct.ampls[pick_idx], self.n_modes)

            kks_var = grp.createVariable("kks","f8", ("nspec",))
            kks_var[:] = self.__pad_zeros(struct.kks[pick_idx], self.n_modes)

            lls_var = grp.createVariable("lls","f8", ("nspec",))
            lls_var[:] = self.__pad_zeros(struct.lls[pick_idx], self.n_modes)

        rootgrp.close()

    class grp_struct(object):
        def __init__(self, c_idx, clat, clon, is_land, analysis = None):
            self.c_idx = c_idx
            self.clat = clat
            self.clon = clon
            self.is_land = is_land

            self.dk = None
            self.dl = None

            self.ampls = None
            self.kks = None
            self.lls = None

            if analysis is not None:
                for key, value in vars(analysis).items():
                    setattr(self, key, value)


    @staticmethod
    def __pad_zeros(lst, n_modes):

        if lst.size < n_modes:
            pad_len = n_modes - lst.size
        else:
            pad_len = 0

        return np.concatenate((lst, np.zeros((pad_len))))



class reader(object):
    """Simple reader class to read HDF5 output written by :class:`src.io.writer`"""

    def __init__(self, fn):
        """
        Parameters
        ----------
        fn : str
            filename of the file to be read
        """
        self.fn = fn

        self.names = {
            "lat": "lat",
            "lon": "lon",
            "recon": "data",
            "ampls": "spec",
            "pmf_sg": "pmf",
        }

    def get_params(self, params):
        """Get the user-defined parameters from the HDF5 file attributes

        Parameters
        ----------
        params : :class:`src.var.params`
            empty instance of the user-defined parameters class to be populated
        """
        file = h5py.File(self.fn)

        for key in file.attrs.keys():
            setattr(params, key, file.attrs[key])

        file.close()

    def read_data(self, idx, name):
        """Read a particular dataset ``name`` from a group ``idx``

        Parameters
        ----------
        idx : str or int
            the group name
        name : str
            the dataset name

        Returns
        -------
        array-like
            the dataset
        """
        file = h5py.File(self.fn)
        dat = file[str(idx)][name][:]
        file.close()

        return np.array(dat)

    def read_all(self, idx, cell):
        """Populate ``cell`` with all datasets in a group ``idx``

        Parameters
        ----------
        idx : int or str
            the group name
        cell : :class:`src.var.topo_cell`
            empty instance of a cell object to be populated
        """
        file = h5py.File(self.fn)

        idx = str(idx)
        for key, value in self.names.items():
            setattr(cell, value, file[idx][key][:])

        file.close()


def fn_gen(params):
    """Automatically generates HDF5 output filename from :class:`src.var.params`.

    Parameters
    ----------
    params : :class:`src.var.params`
        instance of the user parameter class

    Returns
    -------
    str
        automatically generated filename
    """

    if hasattr(params, "fn_tag"):
        tag = params.fn_tag
    else:
        tag = "unnamed"

    if params.enable_merit:
        topo_dat = "merit"
    else:
        topo_dat = "usgs"

    now = datetime.now()

    date = now.strftime("%d%m%y")
    time = now.strftime("%H%M%S")

    ord = ["tag", "topo_dat", "date", "time"]

    fn = ""
    for item in ord:
        fn += locals()[item]
        fn += "_"

    return fn[:-1]
