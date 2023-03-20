import numpy as np
from scipy.spatial import Delaunay


def get_decomposition(topo):
    # Partition lat-lon domain into a number of coarser but regularly spaces points that will form the vertices of the Delaunay triangles.
    xlen = len(topo.lon)
    ylen = len(topo.lat)
    xPoints = np.linspace(0,xlen-1,11)
    yPoints = np.linspace(0,ylen-1,6)

    YY,XX = np.meshgrid(yPoints,xPoints)

    # Now we get the points by index.
    points = np.array([list(item) for item in zip(XX.ravel(), YY.ravel())]).astype('int')

    lat_verts = topo.lat_grid[points[:,1], points[:,0]]
    lon_verts = topo.lon_grid[points[:,1], points[:,0]]

    # Using these indices, we get the list of points in (lon,lat).
    points = np.array([list(item) for item in zip(lon_verts, lat_verts)])

    lats = points[:,1]
    lons = points[:,0]

    # Using scipy spatial, we setup the Delaunay decomposition
    tri = Delaunay(points)

    # Convert the vertices of the simplices to lat-lon values.
    tri.tri_lat_verts = lats[tri.simplices]
    tri.tri_lon_verts = lons[tri.simplices]

    print("Delaunay triangulation object created.")
    print("Number of triangles =", len(tri.tri_lat_verts))

    # Compute the centroid for each vertex.
    tri.tri_clats = tri.tri_lat_verts.sum(axis=1) / 3.0
    tri.tri_clons = tri.tri_lon_verts.sum(axis=1) / 3.0

    return tri