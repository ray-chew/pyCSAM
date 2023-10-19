import matplotlib.pyplot as plt
from   matplotlib.collections import PolyCollection
import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.ticker import (LongitudeFormatter,
                                LatitudeFormatter,
                                LatitudeLocator, LongitudeLocator)

def lat_lon(topo, fs=(10,6), int=1):
    fig = plt.figure(figsize=fs)
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.coastlines()
    im = ax.contourf(topo.lon_grid[::int], topo.lat_grid[::int], topo.topo[::int],
                alpha=0.5,
                transform=ccrs.PlateCarree(),
                cmap='GnBu',
                )

    cax = fig.add_axes([0.99, 0.22, 0.025, 0.55])
    fig.colorbar(im, cax=cax)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.left_labels = False

    gl.xlocator = LongitudeLocator()
    gl.ylocator = LatitudeLocator()
    gl.xformatter = LongitudeFormatter(auto_hide=False)
    gl.yformatter = LatitudeFormatter()

    ax.tick_params(axis="both",
                tickdir='out',
                length=15,
                grid_transform=ccrs.PlateCarree())

    plt.show()



def lat_lon_delaunay(topo, tri, levels, fs=(8,4),   \
                     label_idxs = False, \
                     highlight_indices = [44,45, 88,89, 16,17], \
                     fn = '../output/delaunay.pdf', \
                     output_fig = False, \
                     int = 1, \
                     ):
    plt.figure(figsize=fs)

    im = plt.contourf(topo.lon_grid[::int], topo.lat_grid[::int], topo.topo[::int], levels=levels, cmap='GnBu')
    im.set_clim(0.0, levels[-1])

    points = tri.points

    cbar = plt.colorbar(im,fraction=0.2,pad=0.005, shrink=1.0)

    plt.triplot(points[:,0], points[:,1], tri.simplices, c='C7', lw=0.5)


    plt.plot(points[:,0], points[:,1], 'wo', ms=2.0)
    # plt.plot(tri_clons, tri_clats, 'rx', ms=4.0)

    if label_idxs:
        highlight_indices = np.array(highlight_indices)
        tri_indices = np.arange(len(tri.tri_lat_verts))

        for idx in tri_indices:
            colour = 'C7'
            fw = None
            
            if (idx in highlight_indices) or (idx in highlight_indices+1):
                colour='C3'
                fw = 'bold'
        
            plt.annotate(tri_indices[idx], (tri.tri_clons[idx],tri.tri_clats[idx]), (tri.tri_clons[idx]-0.3,tri.tri_clats[idx]-0.2), c=colour, fontweight=fw)

    plt.xlabel("longitude [deg.]")
    plt.ylabel("latitude [deg.]")
    plt.tight_layout()
    if output_fig: plt.savefig(fn)
    plt.show()


def error_delaunay(topo, tri, fs=(8,4),   \
                     label_idxs = False, \
                     highlight_indices = [44,45, 88,89, 16,17], \
                     fn = '../output/delaunay.pdf', \
                     output_fig = False, \
                     iint = 1, \
                     errors = None, \
                     alpha_max = 0.5,
                  ):
    fig = plt.figure(figsize=fs)
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.coastlines()
    ax.contourf(topo.lon_grid[::iint], topo.lat_grid[::iint], topo.topo[::iint],
                alpha=1.0,
                transform=ccrs.PlateCarree(),
                cmap='binary',
                )
    
    points = tri.points

    from matplotlib.colors import ListedColormap
    cmap = plt.cm.RdYlGn
    my_cmap = cmap(np.arange(cmap.N))


    zeros_len = 30 # must be even
    lcmap_ov2 = cmap.N / 2
    my_cmap[:,-1] = np.concatenate((np.linspace(0, alpha_max, int(lcmap_ov2 - zeros_len/2))[::-1], np.zeros(zeros_len), np.linspace(0, alpha_max, int(lcmap_ov2 - zeros_len/2))))
    my_cmap = ListedColormap(my_cmap)

    im = ax.tripcolor(points[:,0], points[:,1], tri.simplices.copy(), facecolors=errors, edgecolors='k', cmap=my_cmap, alpha=None)

    if label_idxs:
        highlight_indices = np.array(highlight_indices)
        tri_indices = np.arange(len(tri.tri_lat_verts))

        for idx in tri_indices:
            colour = 'C7'
            fw = None
            
            if (idx in highlight_indices) or (idx in highlight_indices+1):
                colour='C3'
                fw = 'bold'
        
            ax.annotate(tri_indices[idx], (tri.tri_clons[idx],tri.tri_clats[idx]), (tri.tri_clons[idx]-0.3,tri.tri_clats[idx]-0.2), c=colour, fontweight=fw)

    cax = fig.add_axes([0.99, 0.22, 0.025, 0.55])
    fig.colorbar(im, cax=cax)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=2, color='gray', alpha=0.0, linestyle='--')
    gl.top_labels = False
    gl.left_labels = False

    gl.xlocator = LongitudeLocator()
    gl.ylocator = LatitudeLocator()
    gl.xformatter = LongitudeFormatter(auto_hide=False)
    gl.yformatter = LatitudeFormatter()

    ax.tick_params(axis="both",
                tickdir='out',
                length=15,
                grid_transform=ccrs.PlateCarree())

    
    plt.xlabel("longitude [deg.]")
    plt.ylabel("latitude [deg.]")
    plt.tight_layout()
    if output_fig: plt.savefig(fn)

    plt.show()
    
                     

def lat_lon_icon(topo, triangles, \
                 fs=(10,6), \
                 annotate_idxs = True, \
                 title = "", \
                 set_global = False, \
                 fn = '../output/icon_lam.pdf', \
                 output_fig = False, \
                 **kwargs):
    # Taken from https://docs.dkrz.de/doc/visualization/sw/python/source_code/python-matplotlib-example-unstructured-icon-triangles-plot-python-3.html

    #-- set projection
    projection = ccrs.PlateCarree()

    #-- create figure and axes instances; we need subplots for plot and colorbar
    fig, ax = plt.subplots(figsize=fs, subplot_kw=dict(projection=projection))

    if set_global: ax.set_global()

    im = ax.contourf(topo.lon_grid, topo.lat_grid, topo.topo,
            alpha=1.0,
            transform=ccrs.PlateCarree(),
            cmap='GnBu',
            )

    #-- plot land areas at last to get rid of the contour lines at land
    ax.coastlines(linewidth=0.5, zorder=2)
    ax.gridlines(draw_labels=True, linewidth=0.5, color='dimgray', alpha=0.4, zorder=2)

    #-- plot the title string
    plt.title(title)    

    #-- create polygon/triangle collection
    coll = PolyCollection(triangles, array=None, edgecolors='r', fc='r', alpha=0.2, linewidth=1, transform=ccrs.Geodetic(), zorder=3)
    ax.add_collection(coll)

    print('--> polygon collection done')

    if annotate_idxs:
        ncells = kwargs['ncells']
        clon = kwargs['clon']
        clat = kwargs['clat']
        
        cidx = np.arange(ncells)

        for idx in cidx:
            colour = 'r'
            fw = 2

            plt.annotate(cidx[idx], (clon[idx],clat[idx]), (clon[idx]-0.3,clat[idx]-0.2), c=colour, fontweight=fw)

    #-- maximize and save the PNG file
    if output_fig:
        plt.savefig(fn, bbox_inches='tight',dpi=200)