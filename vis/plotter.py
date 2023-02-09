import numpy as np
import matplotlib.pyplot as plt

class fig_obj(object):

    def __init__(self, fig, nhi, nhj):
        self.nhi = nhi
        self.nhj = nhj
        self.fig = fig


    def phys_panel(self, axs, data, title=""):
        extent = [-data.shape[1]/2., data.shape[1]/2., -data.shape[0]/2., data.shape[0]/2. ]

        im = axs.imshow(data, extent=extent, origin='lower', aspect='equal')
        axs.set_title(title)
        self.fig.colorbar(im, ax=axs,fraction=0.2, pad=0.04, shrink=0.7)

        return axs


    def freq_panel(self, axs, ampls, nhi=None, nhj=None, title="Power spectrum"):
        if ((nhi is None) and (nhj is None)):
            nhi = self.nhi
            nhj = self.nhj

        im = axs.pcolormesh(np.abs(ampls), edgecolors='k', cmap='Greys')
        self.fig.colorbar(im,ax=axs,fraction=0.2, pad=0.04, shrink=0.7)

        m_j = np.arange(-nhj/2+1,nhj/2+1)
        ylocs = np.arange(.5,nhj+.5,1.0)

        axs.set_yticks(ylocs, m_j)
        axs.set_xticks(ylocs, np.arange(0,nhj))
        axs.set_title(title)
        axs.set_xlabel(r'$k_n$', fontsize=12)
        axs.set_ylabel(r'$l_m$', fontsize=12)
        axs.set_aspect('equal')

        return axs



    
        
