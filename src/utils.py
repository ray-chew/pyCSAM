import numpy as np
import numba as nb
import scipy.signal as signal
import sys

def pick_cell(lat_ref, lon_ref, grid, radius=1.0):
    clat, clon = grid.clat, grid.clon
    index = np.nonzero((np.abs(clat-lat_ref)<=radius) & 
                       (np.abs(clon-lon_ref)<=radius))[0]
    
    if len(index) == 0:
        return pick_cell(lat_ref, lon_ref, grid, radius=2.0*radius)
    else:
        # pick the centre closest to the reference location
        dist = np.abs(clat[index]-lat_ref) + np.abs(clon[index]-lon_ref) 
        ind = np.argmin(dist)

    return index[ind]


def rad2deg(val):
    return np.rad2deg(val)


def isoceles(grid, cell, res=480):
    grid.clon_vertices = np.array([[0-1e-7, np.pi, (2.0 * np.pi)+1e-7],])
    grid.clat_vertices = np.array([[0-1e-7, (2.0 * np.pi)+1e-7, 0-1e-7],])

    cell.lat = np.linspace(0, 2.0 * np.pi, res)
    cell.lon = np.linspace(0, 2.0 * np.pi, res)

    return 0

def delaunay(grid, cell, res_x=480, res_y=480, xmax = 2.0 * np.pi, 
    ymax = 2.0 * np.pi):
    grid.clon_vertices = np.array([[0-1e-7, 0-1e-7, xmax],])
    grid.clat_vertices = np.array([[0-1e-7, ymax, 0-1e-7],])

    cell.lat = np.linspace(0, ymax, res_x)
    cell.lon = np.linspace(0, xmax, res_y)

    return 0


def gen_art_terrain(shp, seed = 555, iters = 1000):
    np.random.seed(seed)
    k = np.random.random (shp)

    dt = 0.1
    for _ in range(iters):
        kp = np.pad(k, ((1,1),(1,1)), mode='wrap')
        kll = kp[:-2,1:-1]
        krr = kp[2:,1:-1]
        ktt = kp[1:-1,2:]
        kbb = kp[1:-1,:-2]
        k = k + dt * (kll + krr + ktt + kbb - 4.0 * k)

    k -= k.mean()
    var = k.max() - k.min()
    k /= 0.5 * var

    return k


class triangle(object):

    def __init__(self, vx, vy):
        # self.x1, self.x2, self.x3 = vx
        # self.y1, self.y2, self.y3 = vy
        vx = np.append(vx, vx[0])
        vy = np.append(vy, vy[0])

        vx = rescale(vx)
        vy = rescale(vy)

        polygon = np.array([list(item) for item in zip(vx, vy)])

        # self.vec_get_mask = np.vectorize(self.get_mask)
        self.vec_get_mask = self.mask_wrapper(polygon)

    # def get_mask(self, x, y):

    #     x1, x2, x3 = self.x1, self.x2, self.x3
    #     y1, y2, y3 = self.y1, self.y2, self.y3

    #     e1 = self.vector(x1,y1,x2,y2) # edge 1
    #     e2 = self.vector(x2,y2,x3,y3) # edge 2
    #     e3 = self.vector(x3,y3,x1,y1) # edge 3
        
    #     p2e1 = self.vector(x,y,x1,y1) # point to edge 1
    #     p2e2 = self.vector(x,y,x2,y2) # point to edge 2
    #     p2e3 = self.vector(x,y,x3,y3) # point to edge 3
        
    #     c1 = np.cross(e1,p2e1)  # cross product 1
    #     c2 = np.cross(e2,p2e2)  # cross product 2
    #     c3 = np.cross(e3,p2e3)  # cross product 3
        
    #     return np.sign(c1) == np.sign(c2) == np.sign(c3)

    # @staticmethod
    # def vector(x1,y1,x2,y2):
    #     return [x2-x1, y2-y1]
    
    def mask_wrapper(self, polygon):
        return lambda p : self.is_inside_sm(p, polygon)


    # Define function that computes whether a point is in a polygon, and rescales the lat-lon grid to a local coordinate between [0,1].
    # Taken from: https://github.com/sasamil/PointInPolygon_Py/blob/master/pointInside.py
    @staticmethod
    @nb.njit(cache=True)
    def is_inside_sm(point, polygon):
        length = len(polygon)-1
        dy2 = point[1] - polygon[0][1]
        intersections = 0
        ii = 0
        jj = 1

        while ii<length:
            dy  = dy2
            dy2 = point[1] - polygon[jj][1]

            # consider only lines which are not completely above/bellow/right from the point
            if dy*dy2 <= 0.0 and (point[0] >= polygon[ii][0] or point[0] >= polygon[jj][0]):

                # non-horizontal line
                if dy<0 or dy2<0:
                    F = dy*(polygon[jj][0] - polygon[ii][0])/(dy-dy2) + polygon[ii][0]

                    if point[0] > F: # if line is left from the point - the ray moving towards left, will intersect it
                        intersections += 1
                    elif point[0] == F: # point on line
                        return 1

                # point on upper peak (dy2=dx2=0) or horizontal line (dy=dy2=0 and dx*dx2<=0)
                elif dy2==0 and (point[0]==polygon[jj][0] or (dy==0 and (point[0]-polygon[ii][0])*(point[0]-polygon[jj][0])<=0)):
                    return 1

            ii = jj
            jj += 1

        #print 'intersections =', intersections
        return intersections & 1  
    

def rescale(arr):
    arr -= arr.min()
    arr /= arr.max()
    
    return arr


# ref: https://github.com/bosswissam/pysize
def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def get_lat_lon_segments(lat_verts, lon_verts, cell, topo, triangle, rect=False, filtered=True, padding=0):    
    lat_max = get_closest_idx(lat_verts.max(), topo.lat) + padding
    lat_min = get_closest_idx(lat_verts.min(), topo.lat) - padding

    lon_max = get_closest_idx(lon_verts.max(), topo.lon) + padding
    lon_min = get_closest_idx(lon_verts.min(), topo.lon) - padding
    
    cell.lat = np.copy(topo.lat[lat_min : lat_max])
    cell.lon = np.copy(topo.lon[lon_min : lon_max])

    lon_origin = cell.lon[0]
    lat_origin = cell.lat[0]

    cell.lat = latlon2m(cell.lat, lon_origin, latlon='lat')
    cell.lon = latlon2m(cell.lon, lat_origin, latlon='lon')

    cell.wlat = np.diff(cell.lat).max()
    cell.wlon = np.diff(cell.lon).max()
    
    # cell.wlat = np.diff(latlon2m(cell.lat, lon_origin, latlon='lat')).max()
    # cell.wlon = np.diff(latlon2m(cell.lon, lat_origin, latlon='lon')).max()

    # cell.lat = 

    # cell.lat = rescale(cell.lat) * 2.0 * np.pi
    # cell.lon = rescale(cell.lon) * 2.0 * np.pi
    
    cell.topo = np.copy(topo.topo[lat_min:lat_max, lon_min:lon_max])

    ampls = np.fft.fft2(cell.topo) 
    ampls /= ampls.size
    wlat = cell.wlat
    wlon = cell.wlon

    kks = np.fft.fftfreq(cell.topo.shape[1])
    lls = np.fft.fftfreq(cell.topo.shape[0])
    
    kkg, llg = np.meshgrid(kks, lls)

    kls = ((2.0 * np.pi * kkg/wlon)**2 + (2.0 * np.pi * llg/wlat)**2)**0.5

    if filtered: ampls *= np.exp(-(kls / (2.0 * np.pi / 5000))**2.0)

    cell.topo = np.fft.ifft2(ampls * ampls.size).real

    cell.gen_mgrids()
    
    if rect:
        cell.get_masked(mask=np.ones_like(cell.topo).astype('bool'))
    else:
        cell.get_masked(triangle=triangle)
    
    cell.topo_m -= cell.topo_m.mean()
                        
def get_closest_idx(val, arr):
    return int(np.argmin(np.abs(arr - val)))


def latlon2m(arr, fix_pt, latlon):
    arr = np.array(arr)
    assert arr.ndim == 1
    origin = arr[0]

    res = np.zeros_like(arr)
    res[0] = 0.0

    for cnt, idx in enumerate(range(1,len(arr))):
        cnt += 1
        if latlon == 'lat':
            res[cnt] = latlon2m_converter(fix_pt, fix_pt, origin, arr[idx])
        elif latlon == 'lon':
            res[cnt] = latlon2m_converter(origin, arr[idx], fix_pt, fix_pt)
        else:
            assert 0

    return res * 1000
        
# https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude
def latlon2m_converter(lon1, lon2, lat1, lat2):
    # Approximate radius of earth in km
    R = 6373.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance


# ref: https://gist.github.com/meowklaski/4bda7c86c6168f3557657d5fb0b5395a
def sliding_window_view(arr, window_shape, steps):
    """ 
    Produce a view from a sliding, striding window over `arr`.
    The window is only placed in 'valid' positions - no overlapping
    over the boundary.

    Parameters
    ----------
    arr : numpy.ndarray, shape=(...,[x, (...), z])
        The array to slide the window over.
    window_shape : Sequence[int]
        The shape of the window to raster: [Wx, (...), Wz],
        determines the length of [x, (...), z]
    steps : Sequence[int]
        The step size used when applying the window
        along the [x, (...), z] directions: [Sx, (...), Sz]

    Returns
    -------
    view of `arr`, shape=([X, (...), Z], ..., [Wx, (...), Wz]), where X = (x - Wx) // Sx + 1

    Note
    -----
    In general, given::

        out = sliding_window_view(arr,
                                    window_shape=[Wx, (...), Wz],
                                    steps=[Sx, (...), Sz])
        out[ix, (...), iz] = arr[..., ix*Sx:ix*Sx+Wx,  (...), iz*Sz:iz*Sz+Wz]

    This function is taken from:
    https://gist.github.com/meowklaski/4bda7c86c6168f3557657d5fb0b5395a

    Example
    --------
    >>> import numpy as np
    >>> x = np.arange(9).reshape(3,3)
    >>> x
    array([[0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]])
    >>> y = sliding_window_view(x, window_shape=(2, 2), steps=(1, 1))
    >>> y
    array([[[[0, 1],
            [3, 4]],
            [[1, 2],
            [4, 5]]],
        [[[3, 4],
            [6, 7]],
            [[4, 5],
            [7, 8]]]])
    >>> np.shares_memory(x, y)
        True
    # Performing a neural net style 2D conv (correlation)
    # placing a 4x4 filter with stride-1
    >>> data = np.random.rand(10, 3, 16, 16)  # (N, C, H, W)
    >>> filters = np.random.rand(5, 3, 4, 4)  # (F, C, Hf, Wf)
    >>> windowed_data = sliding_window_view(data,
    ...                                     window_shape=(4, 4),
    ...                                     steps=(1, 1))
    >>> conv_out = np.tensordot(filters,
    ...                         windowed_data,
    ...                         axes=[[1,2,3], [3,4,5]])
    # (F, H', W', N) -> (N, F, H', W')
    >>> conv_out = conv_out.transpose([3,0,1,2])

    """
         
    from numpy.lib.stride_tricks import as_strided
    in_shape = np.array(arr.shape[-len(steps):])  # [x, (...), z]
    window_shape = np.array(window_shape)  # [Wx, (...), Wz]
    steps = np.array(steps)  # [Sx, (...), Sz]
    nbytes = arr.strides[-1]  # size (bytes) of an element in `arr`

    # number of per-byte steps to take to fill window
    window_strides = tuple(np.cumprod(arr.shape[:0:-1])[::-1]) + (1,)
    # number of per-byte steps to take to place window
    step_strides = tuple(window_strides[-len(steps):] * steps)
    # number of bytes to step to populate sliding window view
    strides = tuple(int(i) * nbytes for i in step_strides + window_strides)

    outshape = tuple((in_shape - window_shape) // steps + 1)
    # outshape: ([X, (...), Z], ..., [Wx, (...), Wz])
    outshape = outshape + arr.shape[:-len(steps)] + tuple(window_shape)
    return as_strided(arr, shape=outshape, strides=strides, writeable=False)


class taper(object):
    def __init__(self, cell, padding, stencil_typ='OP', scale_fac=1.0, art_dt=0.5, art_it=800):
        stencil = np.zeros((3,3))

        if stencil_typ == 'OP':
            self.stencil = self.stencil_OP(stencil)
        elif stencil_typ == '5pt':
            self.stencil = self.stencil_5pt(stencil)
        elif stencil_typ == 'PK':
            self.stencil = self.stencil_PK(stencil)

        self.stencil *= scale_fac

        self.art_dt = art_dt
        self.art_it = art_it
        self.padding = padding

        self.__apply_mask_padding(cell)

    def __apply_mask_padding(self, cell):
        p0 = cell.mask
        self.p0 = np.pad(p0, (self.padding,self.padding), mode='constant')

        self.p = np.copy(self.p0)

    def do_tapering(self):
        for _ in range(self.art_it):
            # artificial diffusion / Shapiro filter
            self.p = self.p + self.art_dt * signal.convolve2d(self.p, self.stencil, mode='same')

            # resetting of the topography mask
            self.p *= ~self.p0
            self.p += self.p0

        del self.p0


    # https://en.wikipedia.org/wiki/Nine-point_stencil
    # I tried the 5pt stencil but it struggles when art_dt is large.
    # From experience, the most robust stencil is the isotropic Oono-Puri 
    @staticmethod
    def stencil_5pt(stencil):
        stencil[0,1] = 1.0
        stencil[1,0] = 1.0
        stencil[1,2] = 1.0
        stencil[2,1] = 1.0
        stencil[1,1] = -4.0
        return stencil

    @staticmethod
    def stencil_OP(stencil):
        stencil[:,:] = 0.25
        stencil[0,1] = 0.50
        stencil[1,0] = 0.50
        stencil[1,2] = 0.50
        stencil[2,1] = 0.50
        stencil[1,1] = -3.0
        return stencil
    
    @staticmethod
    def stencil_PK(stencil):
        stencil[:,:] = 1.0/6
        stencil[0,1] = 4.0/6
        stencil[1,0] = 4.0/6
        stencil[1,2] = 4.0/6
        stencil[2,1] = 4.0/6
        stencil[1,1] = -20.0/6
        return stencil
    

        