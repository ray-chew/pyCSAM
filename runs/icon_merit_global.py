# %%
import numpy as np

from pycsam.src import io, var, utils
from pycsam.wrappers import interface, diagnostics
from pycsam.vis import cart_plot

# from IPython import get_ipython

# ipython = get_ipython()

# if ipython is not None:
#     ipython.run_line_magic("load_ext", "autoreload")
# else:
#     print(ipython)

# def autoreload():
#     if ipython is not None:
#         ipython.run_line_magic("autoreload", "2")

# from sys import exit

# if __name__ != "__main__":
#     exit(0)


# %%
def do_cell(c_idx,
            grid,
            params,
            reader,
            writer,
            ):
    topo = var.topo_cell()
    lat_verts = grid.clat_vertices[c_idx]
    lon_verts = grid.clon_vertices[c_idx]

    lat_extent = [lat_verts.min() - 1.0,lat_verts.min() - 1.0,lat_verts.max() + 1.0]
    lon_extent = [lon_verts.min() - 1.0,lon_verts.min() - 1.0,lon_verts.max() + 1.0]
    # we only keep the topography that is inside this lat-lon extent.
    lat_verts = np.array(lat_extent)
    lon_verts = np.array(lon_extent)

    params.lat_extent = lat_extent
    params.lon_extent = lon_extent


    reader = reader.read_merit_topo(None, params, is_parallel=True)
    reader.get_topo(topo)
    # reader.close_all()
    topo.topo[np.where(topo.topo < -500.0)] = -500.0

    topo.gen_mgrids()


# %%

    clon = np.array([grid.clon[c_idx]])
    clat = np.array([grid.clat[c_idx]])
    clon_vertices = np.array([grid.clon_vertices[c_idx]])
    clat_vertices = np.array([grid.clat_vertices[c_idx]])

    ncells = 1
    nv = clon_vertices[0].size
    # -- create the triangles
    clon_vertices = np.where(clon_vertices < -180.0, clon_vertices + 360.0, clon_vertices)
    clon_vertices = np.where(clon_vertices > 180.0, clon_vertices - 360.0, clon_vertices)

    triangles = np.zeros((ncells, nv, 2))

    for i in range(0, ncells, 1):
        triangles[i, :, 0] = np.array(clon_vertices[i, :])
        triangles[i, :, 1] = np.array(clat_vertices[i, :])

    print("--> triangles done")

    if params.plot:
        cart_plot.lat_lon_icon(topo, triangles, ncells=ncells, clon=clon, clat=clat)

# %%
    tri_idx = 0
    # initialise cell object
    cell = var.topo_cell()
    tri = var.obj()

    nhi = params.nhi
    nhj = params.nhj

    fa = interface.first_appx(nhi, nhj, params, topo)
    sa = interface.second_appx(nhi, nhj, params, topo, tri)

    dplot = diagnostics.diag_plotter(params, nhi, nhj)


    tri.tri_lon_verts = triangles[:, :, 0]
    tri.tri_lat_verts = triangles[:, :, 1]


    simplex_lat = tri.tri_lat_verts[tri_idx]
    simplex_lon = tri.tri_lon_verts[tri_idx]

    if utils.is_land(cell, simplex_lat, simplex_lon, topo):
        writer.output(c_idx, clat_rad[c_idx], clon_rad[c_idx], 0)
        return
    else:
        is_land = 1

    if params.dfft_first_guess:
        # do tapering
        if params.taper_fa:
            interface.taper_quad(params, simplex_lat, simplex_lon, cell, topo)
        else:
            utils.get_lat_lon_segments(
                simplex_lat, simplex_lon, cell, topo, rect=params.rect
            )  

        dfft_run = interface.get_pmf(nhi, nhj, params.U, params.V)
        ampls_fa, uw_fa, dat_2D_fa, kls_fa = dfft_run.dfft(cell)

        cell_fa = cell

        nhi = len(cell_fa.lon)
        nhj = len(cell_fa.lat)

        sa.nhi = nhi
        sa.nhj = nhj
    else:
        cell_fa, ampls_fa, uw_fa, dat_2D_fa = fa.do(simplex_lat, simplex_lon)


    sols = (cell_fa, ampls_fa, uw_fa, dat_2D_fa)

    v_extent = [dat_2D_fa.min(), dat_2D_fa.max()]

    if params.plot:
        if params.dfft_first_guess:
            dplot.show(
            tri_idx, sols, kls=kls_fa, v_extent=v_extent, dfft_plot=True,
            output_fig=False
        )
        else:
            dplot.show(tri_idx, sols, v_extent=v_extent, output_fig=False)

    if params.recompute_rhs:
        sols, _ = sa.do(tri_idx, ampls_fa)
    else:
        sols = sa.do(tri_idx, ampls_fa)

    cell, ampls_sa, uw_sa, dat_2D_sa = sols
    v_extent = [dat_2D_sa.min(), dat_2D_sa.max()]

    writer.output(c_idx, clat_rad[c_idx], clon_rad[c_idx], is_land, cell.analysis)

    if params.plot:
        if params.dfft_first_guess:
            dplot.show(
            tri_idx, sols, kls=kls_fa, v_extent=v_extent, dfft_plot=True,
            output_fig=False
        )
        else:
            dplot.show(tri_idx, sols, v_extent=v_extent, output_fig=False)

    return 1


def parallel_wrapper(grid, params, reader, writer):
    return lambda ii : do_cell(ii, grid, params, reader, writer)



# %%

# autoreload()
from pycsam.inputs.icon_regional_run import params

# %%
from dask.distributed import Client, progress
import dask

if __name__ == '__main__':
    if params.self_test():
        params.print()

    print(params.path_compact_topo)

    grid = var.grid()

    # read grid
    reader = io.ncdata(padding=params.padding, padding_tol=(60 - params.padding))

    # writer object
    writer = io.nc_writer(params)

    reader.read_dat(params.path_compact_grid, grid)

    clat_rad = np.copy(grid.clat)
    clon_rad = np.copy(grid.clon)

    grid.apply_f(utils.rad2deg)

    n_cells = grid.clat.size

    print(n_cells)

    pw_run = parallel_wrapper(grid, params, reader, writer)

    client = Client(threads_per_worker=2, n_workers=4)

    lazy_results = []

    for c_idx in range(n_cells)[:12]:
        # pw_run(c_idx)
        lazy_result = dask.delayed(pw_run)(c_idx)
        lazy_results.append(lazy_result)

    results = dask.compute(*lazy_results)

    # merit_reader.close_all()

