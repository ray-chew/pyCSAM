import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator, LongitudeLocator)

def lat_lon(topo, fs=(10,6)):
    fig = plt.figure(figsize=fs)
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.coastlines()
    im = ax.contourf(topo.lon_grid, topo.lat_grid, topo.topo,
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
                     fn = 'output/delaunay.pdf', \
                     output_fig = False
                     ):
    plt.figure(figsize=fs)

    im = plt.contourf(topo.lon_grid, topo.lat_grid, topo.topo, levels=levels, cmap='GnBu')
    im.set_clim(0.0, levels[-1])

    points = tri.points

    cbar = plt.colorbar(im,fraction=0.2,pad=0.005, shrink=1.0)
    plt.triplot(points[:,0], points[:,1], tri.simplices, c='C7', lw=0.5)
    plt.plot(points[:,0], points[:,1], 'wo', ms=2.0)
    # plt.plot(tri_clons, tri_clats, 'rx', ms=4.0)

    if label_idxs:
        tri_indices = np.arange(len(tri.tri_lat_verts))

        for idx in tri_indices:
            colour = 'C7'
            fw = None
            
            if idx in highlight_indices:
                colour='C3'
                fw = 'bold'
        
            plt.annotate(tri_indices[idx], (tri.tri_clons[idx],tri.tri_clats[idx]), (tri.tri_clons[idx]-0.3,tri.tri_clats[idx]-0.2), c=colour, fontweight=fw)

    plt.xlabel("longitude [deg.]")
    plt.ylabel("latitude [deg.]")
    plt.tight_layout()
    if output_fig: plt.savefig(fn)
    plt.show()