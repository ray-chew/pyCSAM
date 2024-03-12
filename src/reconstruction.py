import numpy as np

def recon_2D(recons_z, cell):
    """
    Reassembles the vector-like ``recons_z`` into a 2D representation given by the properties of :class:`cell <src.var.topo_cell>`.

    Parameters
    ----------
    recons_z : list
        reconstructed topography from :func:`src.lin_reg.do`
    cell : :class:`src.var.topo_cell`
        instance of the ``cell`` object

    Returns
    -------
    array-like
        2D reconstructed topography, values outside the mask are set to zero.
    """
    lon, lat = cell.lon, cell.lat

    recons_z_2D = np.zeros(np.shape(cell.topo))
    c = 0
    for i in range(len(lat)):
        for j in range(len(lon)):

            if (cell.mask[i,j] == 1):
                recons_z_2D[i,j] = recons_z[c]
                c = c+1
                
    return recons_z_2D