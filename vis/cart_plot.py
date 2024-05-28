"""
Contains functions for regional limited-area plots.

Requires the `cartopy <https://scitools.org.uk/cartopy/docs/latest/>`_ package.
"""

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap
import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.ticker import (
    LongitudeFormatter,
    LatitudeFormatter,
    LatitudeLocator,
    LongitudeLocator,
)


def lat_lon(topo, fs=(10, 6), int=1):
    """
    Does a simple Plate-Carre projection of a lat-lon topography data.

    Parameters
    ----------
    topo : array-like
        2D topography data
    fs : tuple, optional
        figure size, by default (10,6)
    int : int, optional
        for high-resolution datasets, do we only plot every `int` pixel? By default 1, i.e., everything is plotted.
    """

    fig = plt.figure(figsize=fs)
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.coastlines()
    im = ax.contourf(
        topo.lon_grid[::int],
        topo.lat_grid[::int],
        topo.topo[::int],
        alpha=0.5,
        transform=ccrs.PlateCarree(),
        cmap="GnBu",
    )

    cax = fig.add_axes([0.99, 0.22, 0.025, 0.55])
    fig.colorbar(im, cax=cax)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=2,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.top_labels = False
    gl.left_labels = False

    gl.xlocator = LongitudeLocator()
    gl.ylocator = LatitudeLocator()
    gl.xformatter = LongitudeFormatter(auto_hide=False)
    gl.yformatter = LatitudeFormatter()

    ax.text(
        -0.01,
        0.5,
        "latitude",
        va="bottom",
        ha="center",
        rotation="vertical",
        rotation_mode="anchor",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        -0.15,
        "longitude",
        va="bottom",
        ha="center",
        rotation="horizontal",
        rotation_mode="anchor",
        transform=ax.transAxes,
    )

    ax.tick_params(
        axis="both", tickdir="out", length=15, grid_transform=ccrs.PlateCarree()
    )

    plt.show()


def lat_lon_delaunay(
    topo,
    tri,
    levels,
    fs=(8, 4),
    label_idxs=False,
    highlight_indices=[44, 45, 88, 89, 16, 17],
    fn="../output/delaunay.pdf",
    output_fig=False,
    int=1,
    raster=False,
):
    """
    Plots a Plate-Carrée projection of the topography with a Delunay triangulated grid.

    Parameters
    ----------
    topo : array-like
        2D topography data
    tri : :class:`scipy.spatial.qhull.Delaunay`
        instance of the scipy Delaunay triangulation object containing tuples of the three vertice coordinates of a triangle
    levels : list
        user-defined elevation levels for the plot
    fs : tuple, optional
        figure size, by default (8,4)
    """

    plt.figure(figsize=fs)

    im = plt.contourf(
        topo.lon_grid[::int],
        topo.lat_grid[::int],
        topo.topo[::int],
        levels=levels,
        cmap="GnBu",
    )
    im.set_clim(0.0, levels[-1])

    if raster:
        for c in im.collections:
            c.set_rasterized(True)

    points = tri.points

    cbar = plt.colorbar(im, fraction=0.2, pad=0.005, shrink=1.0)

    plt.triplot(points[:, 0], points[:, 1], tri.simplices, c="C7", lw=0.5, alpha=0.7)

    plt.plot(points[:, 0], points[:, 1], "wo", ms=0.0)
    # plt.plot(tri_clons, tri_clats, 'rx', ms=4.0)

    if label_idxs:
        highlight_indices = np.array(highlight_indices)
        tri_indices = np.arange(len(tri.tri_lat_verts))

        for idx in tri_indices:
            colour = "C7"
            fw = None

            if (idx in highlight_indices) or (idx in highlight_indices + 1):
                colour = "C3"
                fw = "bold"

            plt.annotate(
                tri_indices[idx],
                (tri.tri_clons[idx], tri.tri_clats[idx]),
                (tri.tri_clons[idx] - 0.3, tri.tri_clats[idx] - 0.2),
                c=colour,
                fontweight=fw,
                alpha=0.8,
                fontsize=12,
            )

    plt.xlabel("longitude [deg.]")
    plt.ylabel("latitude [deg.]")
    plt.tight_layout()
    if output_fig:
        plt.savefig(fn)
    plt.show()


def error_delaunay(
    topo,
    tri,
    fs=(8, 4),
    label_idxs=False,
    highlight_indices=[44, 45, 88, 89, 16, 17],
    fn="../output/delaunay.pdf",
    output_fig=False,
    iint=1,
    errors=None,
    alpha_max=0.5,
    v_extent=[-25.0, 25.0],
    raster=True,
    fontsize=12,
):
    """
    Plots the Delaunay triangulation of a lat-lon domain with the correponding errors.

    Parameters
    ----------
    topo : array-like
        2D topography data
    tri : :class:`scipy.spatial.qhull.Delaunay` object
        instance of the scipy Delaunay triangulation object containing tuples of the three vertice coordinates of a triangle
    fs : tuple, optional
        figure size, by default (8,4)
    label_idxs : bool, optional
        toggles index labels, by default False
    highlight_indices : list, optional
        toggles highlighting of given indices, by default [44,45, 88,89, 16,17]
    fn : str, optional
        output file name, by default '../output/delaunay.pdf'
    output_fig : bool, optional
        toggles writing of output figure, by default False
    iint : int, optional
        how many data points to skip in plotting the topography, by default 1, i.e., the full resolution is used.
    errors : list, optional
        list of errors computed within each triangle, by default None
    alpha_max : float, optional
        alpha of the error overlay, by default 0.5
    v_extent : list, optional
        vertical extent of the error, by default [-25.0, 25.0]
    raster : bool, optional
        toggles vector or raster output, by default True
    fontsize : int, optional
        fontsize, by default 12
    """
    fig = plt.figure(figsize=fs)
    # ax = plt.axes(projection=ccrs.PlateCarree())
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    ax.coastlines(alpha=0.5)
    im = ax.contourf(
        topo.lon_grid[::iint],
        topo.lat_grid[::iint],
        topo.topo[::iint],
        alpha=1.0,
        transform=ccrs.PlateCarree(),
        cmap="binary",
    )

    if raster:
        for c in im.collections:
            c.set_rasterized(True)

    points = tri.points

    cmap = plt.cm.RdYlGn
    my_cmap = cmap(np.arange(cmap.N))

    zeros_len = 2  # must be even
    lcmap_ov2 = cmap.N / 2
    my_cmap[:, -1] = np.concatenate(
        (
            np.linspace(0, alpha_max, int(lcmap_ov2 - zeros_len / 2))[::-1],
            np.zeros(zeros_len),
            np.linspace(0, alpha_max, int(lcmap_ov2 - zeros_len / 2)),
        )
    )
    my_cmap = ListedColormap(my_cmap)

    im = ax.tripcolor(
        points[:, 0],
        points[:, 1],
        tri.simplices.copy(),
        facecolors=errors,
        edgecolors="k",
        cmap=my_cmap,
        alpha=0.5,
        vmin=v_extent[0],
        vmax=v_extent[1],
        linewidth=0.05,
    )

    if label_idxs:
        highlight_indices = np.array(highlight_indices)
        tri_indices = np.arange(len(tri.tri_clats))

        for idx in tri_indices:
            colour = "C7"
            fw = None

            if (idx in highlight_indices) or (idx in highlight_indices + 1):
                colour = "C0"
                fw = "bold"

            ax.annotate(
                tri_indices[idx],
                (tri.tri_clons[idx], tri.tri_clats[idx]),
                (tri.tri_clons[idx] - 0.3, tri.tri_clats[idx] - 0.2),
                c=colour,
                fontweight=fw,
            )

    cax = fig.add_axes([1.0, 0.228, 0.025, 0.54])
    # cax = fig.add_axes([0.85, 0.1, 0.025, 0.8])
    fig.colorbar(im, cax=cax)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=2,
        color="gray",
        alpha=0.0,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False

    gl.xlocator = LongitudeLocator()
    gl.ylocator = LatitudeLocator()
    gl.xformatter = LongitudeFormatter(auto_hide=False)
    gl.yformatter = LatitudeFormatter()

    ax.tick_params(
        axis="both", tickdir="out", length=15, grid_transform=ccrs.PlateCarree()
    )

    ax.text(
        -0.05,
        0.5,
        "latitude [deg]",
        va="bottom",
        ha="center",
        rotation="vertical",
        rotation_mode="anchor",
        transform=ax.transAxes,
        fontsize=fontsize,
    )
    ax.text(
        0.5,
        -0.1,
        "longitude [deg]",
        va="bottom",
        ha="center",
        rotation="horizontal",
        rotation_mode="anchor",
        transform=ax.transAxes,
        fontsize=fontsize,
    )

    plt.tight_layout()
    if output_fig:
        plt.savefig(fn, bbox_inches="tight", dpi=200)

    plt.show()


def lat_lon_icon(
    topo,
    triangles,
    fs=(10, 6),
    annotate_idxs=True,
    title="",
    set_global=False,
    fn="../output/icon_lam.pdf",
    output_fig=False,
    **kwargs
):
    """
    Plots the topography given an ICON grid.

    Parameters
    ----------
    topo : array-like
        2D topography data
    triangles : list
        list containing tuples of the three vertice coordinates of a triangle

    Note
    ----
    Reference used: https://docs.dkrz.de/doc/visualization/sw/python/source_code/python-matplotlib-example-unstructured-icon-triangles-plot-python-3.html
    """
    # -- set projection
    projection = ccrs.PlateCarree()

    # -- create figure and axes instances; we need subplots for plot and colorbar
    fig, ax = plt.subplots(figsize=fs, subplot_kw=dict(projection=projection))

    if set_global:
        ax.set_global()

    im = ax.contourf(
        topo.lon_grid,
        topo.lat_grid,
        topo.topo,
        alpha=1.0,
        transform=ccrs.PlateCarree(),
        cmap="GnBu",
    )

    # -- plot land areas at last to get rid of the contour lines at land
    ax.coastlines(linewidth=0.5, zorder=2)
    ax.gridlines(draw_labels=True, linewidth=0.5, color="dimgray", alpha=0.4, zorder=2)

    # -- plot the title string
    plt.title(title)

    # -- create polygon/triangle collection
    coll = PolyCollection(
        triangles,
        array=None,
        edgecolors="r",
        fc="r",
        alpha=0.2,
        linewidth=1,
        transform=ccrs.PlateCarree(),
        zorder=3,
    )
    ax.add_collection(coll)

    print("--> polygon collection done")

    if annotate_idxs:
        ncells = kwargs["ncells"]
        clon = kwargs["clon"]
        clat = kwargs["clat"]

        cidx = np.arange(ncells)

        for idx in cidx:
            colour = "r"
            fw = 2

            plt.annotate(
                cidx[idx],
                (clon[idx], clat[idx]),
                (clon[idx] - 0.3, clat[idx] - 0.2),
                c=colour,
                fontweight=fw,
            )

    # -- maximize and save the PNG file
    if output_fig:
        plt.savefig(fn, bbox_inches="tight", dpi=200)
        plt.close()
