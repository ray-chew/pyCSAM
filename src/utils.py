import numpy as np

def pick_cell(lat_ref, lon_ref, grid):
    clat, clon = grid.clat, grid.clon
    index = np.nonzero((np.abs(clat-lat_ref)<=1) & 
                       (np.abs(clon-lon_ref)<=1))[0]
    
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
        self.x1, self.x2, self.x3 = vx
        self.y1, self.y2, self.y3 = vy

        self.vec_get_mask = np.vectorize(self.get_mask)

    def get_mask(self, x, y):

        x1, x2, x3 = self.x1, self.x2, self.x3
        y1, y2, y3 = self.y1, self.y2, self.y3

        e1 = self.vector(x1,y1,x2,y2) # edge 1
        e2 = self.vector(x2,y2,x3,y3) # edge 2
        e3 = self.vector(x3,y3,x1,y1) # edge 3
        
        p2e1 = self.vector(x,y,x1,y1) # point to edge 1
        p2e2 = self.vector(x,y,x2,y2) # point to edge 2
        p2e3 = self.vector(x,y,x3,y3) # point to edge 3
        
        c1 = np.cross(e1,p2e1)  # cross product 1
        c2 = np.cross(e2,p2e2)  # cross product 2
        c3 = np.cross(e3,p2e3)  # cross product 3
        
        return np.sign(c1) == np.sign(c2) == np.sign(c3)

    @staticmethod
    def vector(x1,y1,x2,y2):
        return [x2-x1, y2-y1]

@np.vectorize
def vector(x1,y1,x2,y2):
    return [x2-x1, y2-y1]

@np.vectorize
def get_mask(x,y, vx, vy):
    x1, x2, x3 = vx
    y1, y2, y3 = vy
    e1 = vector(x1,y1,x2,y2) # edge 1
    e2 = vector(x2,y2,x3,y3) # edge 2
    e3 = vector(x3,y3,x1,y1) # edge 3
    
    p2e1 = vector(x,y,x1,y1) # point to edge 1
    p2e2 = vector(x,y,x2,y2) # point to edge 2
    p2e3 = vector(x,y,x3,y3) # point to edge 3
    
    c1 = np.cross(e1,p2e1)  # cross product 1
    c2 = np.cross(e2,p2e2)  # cross product 2
    c3 = np.cross(e3,p2e3)  # cross product 3
    
    return np.sign(c1) == np.sign(c2) == np.sign(c3)