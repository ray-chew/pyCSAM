import numpy as np
import matplotlib.pyplot as plt

class fig_obj(object):

    def __init__(self, fig, nhi, nhj):
        self.nhi = nhi
        self.nhj = nhj
        self.fig = fig


    def phys_panel(self, axs, data, title="", extent=None, xlabel="", ylabel="", v_extent=None):
        if extent is None:
            extent = [-data.shape[1]/2., data.shape[1]/2., -data.shape[0]/2., data.shape[0]/2. ]
        if v_extent is not None:
            vmin, vmax = v_extent[0], v_extent[1]
        else:
            vmin, vmax = None, None

        # conversion from [m] to [km]
        extent = np.array(extent) / 1000.0

        im = axs.imshow(data, extent=extent, origin='lower', aspect='equal', cmap='cividis', vmin=vmin, vmax=vmax)
        axs.set_title(title)
        axs.set_xlabel(xlabel)
        axs.set_ylabel(ylabel)
        self.fig.colorbar(im, ax=axs,fraction=0.2, pad=0.04, shrink=0.5)

        return axs


    def freq_panel(self, axs, ampls, nhi=None, nhj=None, title="Power spectrum"):
        if ((nhi is None) and (nhj is None)):
            nhi = self.nhi
            nhj = self.nhj

        im = axs.pcolormesh(np.abs(ampls), edgecolors='k', cmap='Greys')
        self.fig.colorbar(im,ax=axs,fraction=0.2, pad=0.04, shrink=0.7)

        m_j = np.arange(-nhj/2+1, nhj/2+1)
        ylocs = np.arange(.5, nhj+.5, 1.0)

        m_i = np.arange(0, nhi)
        xlocs = np.arange(.5, nhi+.5, 1.0)

        axs.set_xticks(xlocs, m_i, rotation=-90)
        axs.set_yticks(ylocs, m_j)
        axs.set_title(title)
        axs.set_xlabel(r'$k_n$', fontsize=12)
        axs.set_ylabel(r'$l_m$', fontsize=12)
        axs.set_aspect('equal')

        return axs


    def fft_freq_panel(self, axs, ampls, kks, lls, \
                       title = "FFT power spectrum", \
                       interval = 20, \
                       typ='imag'
                       ):
        xmid = int(len(kks)/2)
        ymid = int(len(lls)/2)

        if typ == 'imag':
            kks = kks[xmid-interval:xmid+interval]
            lls = lls[ymid-interval:ymid+interval]

            ampls = ampls[ymid-interval:ymid+interval,xmid-interval:xmid+interval]
        elif typ == 'real':
            interval = int(2.0 * interval)
            kks = kks[0:interval]
            lls = lls[0:interval]

            ampls = ampls[0:interval,0:interval]


        xlocs = np.linspace(0, len(kks)-1, 5)+0.5
        xlabels = np.linspace(kks[0], kks[-1], 5)

        ylocs = np.linspace(0, len(lls)-1, 5)+0.5
        ylabels = np.linspace(lls[0], lls[-1], 5)

        xlocs = np.around(xlocs, 2)
        xlabels = np.around(xlabels, 2)
        ylocs = np.around(ylocs, 2)
        ylabels = np.around(ylabels, 2)

        im = axs.imshow(np.abs(ampls), cmap='Greys', origin='lower')
        self.fig.colorbar(im,ax=axs,fraction=0.2, pad=0.04, shrink=0.7)
        axs.set_xticks(xlocs, xlabels)
        axs.set_yticks(ylocs, ylabels)
        axs.set_title(title)
        axs.set_xlabel(r'$k_n$', fontsize=12)
        axs.set_ylabel(r'$l_m$', fontsize=12)
        axs.set_aspect('equal')

        return axs



    
        
