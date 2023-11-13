import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class fig_obj(object):

    def __init__(self, fig, nhi, nhj, cbar=True, set_label=True):
        self.nhi = nhi
        self.nhj = nhj
        self.fig = fig
        self.cbar = cbar
        self.set_label = set_label


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

        if self.set_label:
            axs.set_xlabel(xlabel)
            axs.set_ylabel(ylabel)

        if self.cbar:
            self.fig.colorbar(im, ax=axs,fraction=0.2, pad=0.04, shrink=0.5)

        return axs


    def freq_panel(self, axs, ampls, nhi=None, nhj=None, title="Power spectrum", v_extent=None):
        if ((nhi is None) and (nhj is None)):
            nhi = self.nhi
            nhj = self.nhj

        if v_extent is not None:
            vmin, vmax = v_extent[0], v_extent[1]
        else:
            vmin, vmax = None, None

        im = axs.pcolormesh(np.abs(ampls), edgecolor='k', cmap='Greys', vmin=vmin, vmax=vmax)
        if self.cbar:
            self.fig.colorbar(im,ax=axs,fraction=0.2, pad=0.04, shrink=0.7)

        m_j = np.arange(-nhj/2+1, nhj/2+1)
        ylocs = np.arange(.5, nhj+.5, 1.0)

        m_i = np.arange(0, nhi)
        xlocs = np.arange(.5, nhi+.5, 1.0)

        axs.set_xticks(xlocs, m_i, rotation=-90)
        axs.set_yticks(ylocs, m_j)
        axs.set_title(title)
        # axs.grid(which='minor', color='k', linestyle='-', linewidth=1)
        # axs.set_xlabel(r'$k_n \times 2 \pi / L_x$', fontsize=12)
        # axs.set_ylabel(r'$l_m \times 2 \pi / L_y$', fontsize=12)

        if self.set_label:
            axs.set_ylabel(r'$l_m$', fontsize=12)
        
        axs.set_xlabel(r'$k_n$', fontsize=12)
        # axs.set_aspect('equal')

        for label in axs.yaxis.get_ticklabels()[::2]:
            label.set_visible(False)

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
            lls = lls[ymid-interval:ymid+interval]

            interval_2 = int(2.0 * interval)
            kks = kks[0:interval_2]
            # lls = lls[0:interval_2]

            ampls = ampls[ymid-interval:ymid+interval,0:interval_2]
            # ampls = ampls[0:interval_2,0:interval_2]


        xlocs = np.linspace(0, len(kks)-1, 5)+0.5
        xlabels = np.linspace(kks[0], kks[-1], 5)

        ylocs = np.linspace(0, len(lls)-1, 5)+0.5
        ylabels = np.linspace(lls[0], lls[-1], 5)

        xlocs = np.around(xlocs, 2)
        xlabels = np.around(xlabels, 2)
        ylocs = np.around(ylocs, 2)
        ylabels = np.around(ylabels, 2)

        im = axs.imshow(np.abs(ampls), cmap='Greys', origin='lower')
        if self.cbar:
            self.fig.colorbar(im,ax=axs,fraction=0.2, pad=0.04, shrink=0.7)
        axs.set_xticks(xlocs, xlabels)
        axs.set_yticks(ylocs, ylabels)
        axs.set_title(title)

        if self.set_label:
            axs.set_xlabel(r'$k$ [m$^{-1}$]', fontsize=12)
            axs.set_ylabel(r'$l$ [m$^{-1}$]', fontsize=12)
        if typ == 'imag': axs.set_aspect('equal')

        return axs



def error_bar_plot( idx_name,
                    pmf_diff,
                    params,
                    comparison = None,
                    title="",
                    gen_title=False,
                    output_fig=False,
                    fn="../output/error_plot.pdf",
                    ylim=[-100,100]
                    ):

    data = pd.DataFrame(pmf_diff,index=idx_name, columns=['values'])
    fig, (ax1) = plt.subplots(1,1,sharex=True,
                         figsize=(10.0,6.0))

    if comparison is not None:
        comp_data = pd.DataFrame(comparison, index=idx_name, columns=['values'])

        comp_data['values'].plot(kind='bar', width=1.0, edgecolor='black', color=(comp_data['values'] > 0).map({True: 'C7', False: 'C7'}))

    true_col = 'g'
    false_col = 'C4' if params.dfft_first_guess else 'r'

    data['values'].plot(kind='bar', width=1.0, edgecolor='black', color=(data['values'] > 0).map({True: true_col, False: false_col}))

    plt.grid()

    plt.xlabel("grid idx")
    plt.ylabel("percentage rel. pmf diff")

    avg_err = np.abs(pmf_diff).mean()
    err_input = np.around(avg_err,2)

    if params.dfft_first_guess:
        spec_dom = "(from FFT)"
        fg_tag = 'FFT' 
    else:
        spec_dom = "(%i x %i)" %(params.nhi,params.nhj)
        fg_tag = 'FF'
        
    if params.refine:
        rfn_tag = ' + ext.'
    else:
        rfn_tag = ''

    if gen_title: title = fg_tag + '+FF' + ' ' + rfn_tag + ' avg err: ' + str(err_input)

    plt.title(title, fontsize=12, pad=-10)
    plt.ylim(ylim)
    plt.tight_layout()

    if output_fig: plt.savefig('../output/'+fn+'_poster.pdf')
    plt.show()


def error_bar_split_plot(errs, lbls, bs, ts, ts_ticks,
                         fs=(3.5,3.5), 
                         title="", 
                         output_fig=False, 
                         fn='output/errors.pdf'
                         ):
    
    errs = [np.around(err,2) for err in errs]
    print(errs)

    XX = pd.Series(errs,index=lbls)
    _, (ax1,ax2) = plt.subplots(2,1,sharex=True,
                            figsize=fs)
    ax1.spines['bottom'].set_visible(False)
    ax1.tick_params(axis='x',which='both',bottom=False)
    ax2.spines['top'].set_visible(False)

    ax2.set_ylim(0,bs)
    ax1.set_ylim(ts[0],ts[1])
    ax1.set_yticks(ts_ticks)

    bars1 = ax1.bar(XX.index, XX.values, color=('C0'))
    bars2 = ax2.bar(XX.index, XX.values, color=('C0', 'C1', 'C2', 'r'))
    ax1.bar_label(bars1, padding=3)
    ax2.bar_label(bars2, padding=3)

    for tick in ax2.get_xticklabels():
        tick.set_rotation(0)
    d = .015  
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)      
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax2.transAxes)  
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    for b1, b2 in zip(bars1, bars2):
        posx = b2.get_x() + b2.get_width()/2.
        if b2.get_height() > bs:
            ax2.plot((posx-3*d, posx+3*d), (1 - d, 1 + d), color='k', clip_on=False,
                    transform=ax2.get_xaxis_transform())
        if b1.get_height() > ts[0]:
            ax1.plot((posx-3*d, posx+3*d), (- d, + d), color='k', clip_on=False,
                    transform=ax1.get_xaxis_transform())
            
    plt.title(title, fontsize=18, pad=10)
    plt.tight_layout()
    if output_fig: plt.savefig(fn)
    plt.show()


def error_bar_abs_plot(errs, lbls,
                         fs=(3.5,3.5), 
                         title="", 
                         output_fig=False, 
                         fn='output/errors.pdf',
                         color=None,
                         ylims=None,
                         fontsize=10
                         ):
    
    errs = [np.around(err,2) for err in errs]
    print(errs)

    XX = pd.Series(errs,index=lbls)
    _, (ax1) = plt.subplots(1,1,sharex=True,
                            figsize=fs)
    # ax1.spines['bottom'].set_visible(False)
    # ax1.tick_params(axis='x',which='both',bottom=False)

    bar1 = ax1.bar(XX.index, XX.values, color=color)
    ax1.bar_label(bar1, padding=3)

    if ylims is not None:
        ax1.set_ylim([ylims[0],ylims[1]])

    plt.title(title, fontsize=fontsize, pad=10)
    plt.tight_layout()
    if output_fig: plt.savefig(fn, bbox_inches="tight")
    plt.show()