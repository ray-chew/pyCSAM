import numpy as np

def recon_2D(recons_z, cell):
    lon, lat = cell.lon, cell.lat

    recons_z_2D = np.zeros(np.shape(cell.topo))
    c = 0
    for i in range(len(lat)):
        for j in range(len(lon)):

            if (cell.mask[i,j] == 1):
                recons_z_2D[i,j] = recons_z[c]
                c = c+1
                
    return recons_z_2D