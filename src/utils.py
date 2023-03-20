import numpy as np
import numba as nb
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
    return val*(180/np.pi)


def isoceles(grid, cell, res=480):
    grid.clon_vertices = np.array([[0-1e-7, np.pi, (2.0 * np.pi)+1e-7],])
    grid.clat_vertices = np.array([[0-1e-7, (2.0 * np.pi)+1e-7, 0-1e-7],])

    cell.lat = np.linspace(0, 2.0 * np.pi, res)
    cell.lon = np.linspace(0, 2.0 * np.pi, res)

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