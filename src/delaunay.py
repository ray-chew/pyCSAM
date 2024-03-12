import numpy as np
from scipy.spatial import Delaunay
from src import utils, var

def get_decomposition(topo, xnp=11, ynp=6, padding = 0):
    """
    Partitions a lat-lon domain into a number of coarser but regularly spaced points that comprises the vertices of the Delaunay triangles.

    Parameters
    ----------
    topo : array-like
        2D topography data
    xnp : int, optional
        number of points in the first horizontal direction, by default 11
    ynp : int, optional
        number of points in the second horizontal direction, by default 6
    padding : int, optional
        number of grid points to include as a boundary (padded) region, by default 0

    Returns
    -------
    :class:`scipy.spatial.qhull.Delaunay` instance
        scipy Delaunary triangulation instance
    """

    xlen = len(topo.lon) - padding
    ylen = len(topo.lat) - padding
    xPoints = np.linspace(padding,xlen-1,xnp)
    yPoints = np.linspace(padding,ylen-1,ynp)

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


def get_land_cells(tri, topo, height_tol=0.5, percent_tol=0.95):
    """
    Land cell selector based on how much of a grid cell contains topography of a certain elevation.

    Parameters
    ----------
    tri : :class:`scipy.spatial.qhull.Delaunay` instance
        scipy Delaunay triangulation instance containing tuples of the three vertice coordinates of a triangle
    topo : array-like
        2D topographic data
    height_tol : float, optional
        elevation above `height_tol` are considered as land, by default 0.5 [m]
    percent_tol : float, optional
        cut-off percentage of topography in the given grid cell below `height_tol`. By default 0.95, i.e., at least 5% of the grid cell has to be above `heigh_tol` to be considered a land cell.

    Returns
    -------
    list
        list of land cell indices
    """
    rect_set = []
    n_tri = len(tri.tri_lat_verts)

    for tri_idx in range(n_tri)[::2]:
        cell = var.topo_cell()

        print("computing idx:", tri_idx)

        simplex_lat = tri.tri_lat_verts[tri_idx]
        simplex_lon = tri.tri_lon_verts[tri_idx]

        utils.get_lat_lon_segments(simplex_lat, simplex_lon, cell, topo, load_topo=True, filtered=False)

        if not (((cell.topo <= height_tol).sum() / cell.topo.size) > percent_tol):
            rect_set.append(tri_idx)

    return rect_set