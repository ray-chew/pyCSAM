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
    
    print(c_idx)

    topo = var.topo_cell()

    lat_verts = grid.clat_vertices[c_idx]
    lon_verts = grid.clon_vertices[c_idx]

    # if ( (lon_verts.max() - lon_verts.min()) > 180.0 ):
    #     lon_verts[np.argmin(lon_verts)] += 360.0

    # clon = utils.rescale(grid.clon[c_idx], rng=[lon_verts.min(),lon_verts.max()])
    # clat = utils.rescale(grid.clat[c_idx], rng=[lat_verts.min(),lat_verts.max()])

    # check = utils.gen_triangle(lon_verts, lat_verts)

    # print("is center in triangle:", check.vec_get_mask((clon, clat)))

    # lat_expand = 0.0
    # lat_extent = [lat_verts.min() - lat_expand,lat_verts.min() - lat_expand,lat_verts.max() + lat_expand]

    # lon_expand = 0.0
    # lon_extent = [lon_verts.min() - lon_expand,lon_verts.min() - lon_expand,lon_verts.max() + lon_expand]

    # lat_extent = lat_verts
    # lon_extent = lon_verts
    # we only keep the topography that is inside this lat-lon extent.

    lat_extent, lon_extent = utils.handle_latlon_expansion(lat_verts, lon_verts)

    # lat_verts = np.array(lat_verts)
    # lon_verts = np.array(lon_verts)
    lat_verts, lon_verts = utils.handle_latlon_expansion(lat_verts, lon_verts, lat_expand = 0.0, lon_expand = 0.0)

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
    # clon = np.array([clon])
    # clat = np.array([clat])
    # clon_vertices = np.array([grid.clon_vertices[c_idx]])
    # clat_vertices = np.array([grid.clat_vertices[c_idx]])
    clon_vertices = np.array([lon_verts])
    clat_vertices = np.array([lat_verts])


    ncells = 1
    nv = clon_vertices[0].size
    # -- create the triangles
    # clon_vertices = np.where(clon_vertices < -180.0, clon_vertices + 360.0, clon_vertices)
    # clon_vertices = np.where(clon_vertices > 180.0, clon_vertices - 360.0, clon_vertices)

    # if ( (clon_vertices.max() - clon_vertices.min()) > 180.0 ):
    if reader.split_EW:
        clon_vertices[clon_vertices < 0.0] += 360.0


    triangles = np.zeros((ncells, nv, 2))

    for i in range(0, ncells, 1):
        triangles[i, :, 0] = np.array(clon_vertices[i, :])
        triangles[i, :, 1] = np.array(clat_vertices[i, :])

    if params.plot or params.plot_output:

        output_fn = params.path_output + str(c_idx) + ".png"
        cart_plot.lat_lon_icon(topo, triangles, ncells=ncells, clon=clon, clat=clat, title=c_idx, fn = output_fn, output_fig = True)

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
    dplot.output_dir = params.path_output


    tri.tri_lon_verts = triangles[:, :, 0]
    tri.tri_lat_verts = triangles[:, :, 1]

    simplex_lat = tri.tri_lat_verts[tri_idx]
    simplex_lon = tri.tri_lon_verts[tri_idx]

    if not utils.is_land(cell, simplex_lat, simplex_lon, topo):
        # writer.output(c_idx, clat_rad[c_idx], clon_rad[c_idx], 0)
        print("--> skipping ocean cell")
        return writer.grp_struct(c_idx, clat_rad[c_idx], clon_rad[c_idx], 0)
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
            dplot.show(c_idx, sols, v_extent=v_extent, output_fig=False)

    if params.recompute_rhs:
        sols, _ = sa.do(tri_idx, ampls_fa)
    else:
        sols = sa.do(tri_idx, ampls_fa)

    cell, ampls_sa, uw_sa, dat_2D_sa = sols
    v_extent = [dat_2D_sa.min(), dat_2D_sa.max()]

    # writer.output(c_idx, clat_rad[c_idx], clon_rad[c_idx], is_land, cell.analysis)
    result = writer.grp_struct(c_idx, clat_rad[c_idx], clon_rad[c_idx], is_land, cell.analysis)

    if params.plot:
        if params.dfft_first_guess:
            dplot.show(
            tri_idx, sols, kls=kls_fa, v_extent=v_extent, dfft_plot=True,
            output_fig=False
        )
        else:
            dplot.show(c_idx, sols, v_extent=v_extent, output_fig=False)

    print("--> analysis done")

    return result


def parallel_wrapper(grid, params, reader, writer):
    return lambda ii : do_cell(ii, grid, params, reader, writer)



# %%

# autoreload()
from pycsam.inputs.icon_global_run import params

from dask.distributed import Client
# import dask.bag as db
import dask

# dask.config.set(scheduler='synchronous') 

if __name__ == '__main__':
    if params.self_test():
        params.print()

    grid = var.grid()

    # read grid
    reader = io.ncdata(padding=params.padding, padding_tol=(60 - params.padding))
    # reader.read_dat(params.path_compact_grid, grid)
    reader.read_dat(params.path_icon_grid, grid)

    clat_rad = np.copy(grid.clat)
    clon_rad = np.copy(grid.clon)

    grid.apply_f(utils.rad2deg)

    n_cells = grid.clat.size

    # NetCDF-4 reader does not work well with multithreading
    # Use only 1 thread per worker! (At least on my laptop)
    client = Client(threads_per_worker=1, n_workers=2)

    print(n_cells)

    chunk_sz = 10
    chunk_start = 20400
    for chunk in range(chunk_start, n_cells, chunk_sz):
    # writer object
        sfx = "_" + str(chunk+chunk_sz)
        writer = io.nc_writer(params, sfx)

        pw_run = parallel_wrapper(grid, params, reader, writer)

        lazy_results = []

        # with ProgressBar():
        #     b = db.from_sequence(range(chunk), npartitions=100)
        #     results = b.map(pw_run)
        #     results = results.compute()
        if chunk+chunk_sz > n_cells:
            chunk_end = n_cells
        else:
            chunk_end = chunk+chunk_sz

        for c_idx in range(chunk, chunk_end):
            # pw_run(c_idx)
            lazy_result = dask.delayed(pw_run)(c_idx)
            lazy_results.append(lazy_result)

        results = dask.compute(*lazy_results)

        for item in results:
            writer.duplicate(item.c_idx, item)
