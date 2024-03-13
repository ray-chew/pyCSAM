"""
Diagnostic wrapper module to ease setting up the CSAM building blocks
"""

import numpy as np
from src import physics
from vis import plotter
from copy import deepcopy

import matplotlib.pyplot as plt

class delaunay_metrics(object):
    """Helper class for evaluation of the CSAM on a Delaunay triangulated domain.

    """
    def __init__(self, params, tri, writer=None):
        """

        Parameters
        ----------
        params : :class:`src.var.params`
            instance of the user-defined parameter class
        tri : :class:`scipy.spatial.qhull.Delaunay`
            instance of the scipy Delaunay triangulation class
        writer : :class:`src.io.writer`, optional
            metric will be written to a HDF5 file if writer object is provided, by default None
        """
        self.params = params
        self.tri = tri

        self.pmf_diff = []
        self.pmf_refs = []
        self.pmf_sums = []
        self.pmf_fas  = []
        self.pmf_ssums= []
        self.idx_name = []

        self.writer = writer

    def update_quad(self, idx, uw_ref, uw_fa):
        """Store the computed idealised pseudo-momentum fluxes on a quadrilateral grid, i.e., the reference grid.

        Parameters
        ----------
        idx : str or int
            index of the cell
        uw_ref : array-like
            2D array the size of a dense (truncated) spectral space containing the reference idealised pseudo-momentum fluxes
        uw_fa : array-like
            2D array the size of a dense (truncated) spectral space containing the first-approximation's idealised pseudo-momentum fluxes
        """
        self.uw_ref = uw_ref.sum()
        self.uw_fa = uw_fa.sum()

        self.idx_name.append(idx)
        self.pmf_refs.append(self.uw_ref)
        self.pmf_fas.append(self.uw_fa)


    def get_rel_err(self, triangle_pair):
        """Method to get the relative error explicitly before :func:`runs.diagnostics.delaunay_metrics.end` is called.

        Parameters
        ----------
        triangle_pair : list
            a list containing the index pair in ``int`` for the Delaunay triangles corresponding to a quadrilateral grid cell

        Returns
        -------
        float
            the relative error of the CSAM on the Delaunay triangles against the FFT-computed reference
        """
        self.update_pair(triangle_pair, store_error=False)
        self.rel_err = self.__get_rel_diff(self.uw_sum, self.uw_ref)

        return self.rel_err


    def update_pair(self, triangle_pair, store_error=True):
        """Update metric computation instance with the data from the newly computed triangle pair

        Parameters
        ----------
        triangle_pair : list
            a list containing the index pair in ``int`` for the Delaunay triangles corresponding to a quadrilateral grid cell
        store_error : bool, optional
            keep a list of the errors for each triangle pair, by default True. Otherwise, the errors are discarded and only the average error is stored.
        """
        for triangle in triangle_pair:
            assert hasattr(triangle, 'analysis'), "triangle has no analysis object."

        self.t0 = triangle_pair[0]
        self.t1 = triangle_pair[1]

        self.uw_sum = self.__get_pmf_sum()
        self.uw_spec_sum = self.__get_pmf_spec_sum()

        if store_error:
            self.pmf_sums.append(self.uw_sum)
            self.pmf_ssums.append(self.uw_spec_sum)


    def __get_pmf_sum(self):
        self.uw_0 = self.t0.uw.sum()
        self.uw_1 = self.t1.uw.sum()

        return self.uw_0 + self.uw_1
        

    def __get_pmf_spec_sum(self):
        """Compute the idealised pseudo-momentum fluxes from the sum of the spectra
        """
        self.ampls_0 = self.t0.analysis.ampls
        self.ampls_1 = self.t1.analysis.ampls
        self.ampls_sum = (self.ampls_0 + self.ampls_1)

        # consider replacing deepcopy with copy method.
        analysis_sum = deepcopy(self.t0.analysis)
        analysis_sum.ampls = self.ampls_sum

        ideal = physics.ideal_pmf(U=self.params.U, V=self.params.V)

        return 0.5 * ideal.compute_uw_pmf(analysis_sum)


    def __repr__(self):
        """Redefines what printing the class instance does
        """
        errs = [self.uw_ref, self.uw_fa, self.uw_sum, self.uw_spec_sum]
        errs = [str(err) for err in errs]
        uw_strs = str(self.uw_0) + ', ' +  str(self.uw_1)
        err_strs = ', '.join(errs)

        return uw_strs + '\n' + err_strs + '\n'
    
    def __str__(self):
        return repr(self)
    

    def end(self, verbose=False):
        """Ends the metric computation

        Parameters
        ----------
        verbose : bool, optional
            prints the average errors computed, by default False
        """
        self.__gen_percentage_errs()
        self.__gen_regional_errs()

        if self.writer is not None:
            self.__write()

        if verbose:
            print(np.abs(self.max_errs).mean(), np.abs(self.rel_errs).mean())


    def __write(self):
        """Writes a HDF5 output if a writer class is provided in the initialisation of the class instance
        """
        assert self.writer is not None

        self.writer.populate('decomposition', 'pmf_refs', self.pmf_refs)
        self.writer.populate('decomposition', 'pmf_fas', self.pmf_fas)
        self.writer.populate('decomposition', 'pmf_sums', self.pmf_sums)
        self.writer.populate('decomposition', 'pmf_ssums', self.pmf_ssums)

    def __gen_percentage_errs(self):
        """Computes the relative and maximum errors in percentage
        """
        max_idx = np.argmax(np.abs(self.pmf_refs))
        self.max_errs = self.__get_max_diff(self.pmf_sums, self.pmf_refs, np.array(self.pmf_refs[max_idx]))
        self.rel_errs = self.__get_rel_diff(self.pmf_sums, self.pmf_refs)

        self.max_errs = np.array(self.max_errs) * 100
        self.rel_errs = np.array(self.rel_errs) * 100

    def __gen_regional_errs(self):
        """Computes the relative and maximum errors distributed over the Delaunay triangulation region
        """
        assert hasattr(self, 'max_errs')
        assert hasattr(self, 'rel_errs')

        self.reg_max_errs = self.__get_regional_errs(self.tri, self.max_errs)
        self.reg_rel_errs = self.__get_regional_errs(self.tri, self.rel_errs)

    def __get_regional_errs(self, tri, err):
        """Assigns the (relative or maximum) errors to the corresponding grid cells
        """
        errors = np.zeros((len(tri.simplices)))
        errors[:] = np.nan
        errors[self.params.rect_set] = err
        errors[np.array(self.params.rect_set)+1] = err

        return errors

    @staticmethod
    def __get_rel_diff(arr, ref):
        arr = np.array(arr)
        ref = np.array(ref)

        return arr / ref - 1.0

    @staticmethod
    def __get_max_diff(arr, ref, max):
        arr = np.array(arr)
        ref = np.array(ref)

        return (arr - ref) / max
    

class diag_plotter(object):
    """Helper class to plot CSAM-computed data 
    """
    def __init__(self, params, nhi, nhj):
        """

        Parameters
        ----------
        params : :class:`src.var.params`
            instance of the user-defined parameter class
        nhi : int
            number of harmonics in the first horizontal direction
        nhj : int
            number of harmonics in the second horizontal direction
        """
        self.params = params
        self.nhi = nhi
        self.nhj = nhj

        self.output_dir = "../manuscript/"

    def show(self, rect_idx, sols, kls=None, v_extent=None, dfft_plot=False, output_fig=True, fs = (14.0,4.0), ir_args=None, fn=None):
        """Plots the data

        Parameters
        ----------
        rect_idx : int
            index of the quadrilateral grid cell
        sols : tuple
            contains the data for plotting: 
               | (:class:`src.var.topo_cell` instance, 
               | computed CSAM spectrum, 
               | computed idealised pseudo-momentum fluxes, 
               | the reconstructed physical data)

            ``sols`` is the tuple returned by :func:`runs.interface.first_appx.do` and :func:`runs.interface.second_appx.do`
        kls : list, optional
            list of size 2, each element is a vector containing the (k,l)-wavenumbers, by default None. Only required to plot FFT spectra.
        v_extent : list, optional
            ``[z_min, z_max]`` the vertical extent of the physical reconstruction, by default None
        dfft_plot : bool, optional
            toggles whether a spectrum is the full FFT spectral space or the dense truncated CSAM spectrum, By default False, i.e. plot CSAM spectrum.
        output_fig : bool, optional
            toggles writing figure output, by default True
        fs : tuple, optional
            figure size, by default (14.0,4.0)
        ir_args : list, optional
            additional user-defined arguments: 
               | [title of the physical reconstruction panel,
               | title of the power spectrum panel,
               | title of the idealised pseudo-momentum flux panel,
               | vertical extent of the power spectrum,
               | vertical extent of the idealised pseudo-momentum flux spectrum]
            
            By default None
        fn : str, optional
            output filename, by default None
        """

        cell, ampls, uw, dat_2D = sols

        if v_extent is None:
            v_extent = [dat_2D.min(), dat_2D.max()]

        if ir_args is None:
            if type(rect_idx) is int: 
                idxs_tag = "Cell %i" %rect_idx
                tag = "CSAM"
                fn = "plots_CSAM_%i" %rect_idx
            elif len(rect_idx) == 2:
                idxs_tag = "(%i,%i)" %(rect_idx[0],rect_idx[1])
                tag = "FFT" if dfft_plot else "FA LSFF"
                fn = "plots_%s_%i_%i" %(tag.replace(" ","_"), rect_idx[0], rect_idx[1])
            else:
                idxs_tag = ""
                tag = ""
                fn = "plots_%s" %str(rect_idx)

            t1 = '%s: %s reconstruction' %(idxs_tag,tag)
            if dfft_plot:
                t2 = "ref. power spectrum"
                t3 = "ref. PMF spectrum"
            else:
                t2 = "approx. power spectrum"
                t3 = "approx. PMF spectrum"
            
            freq_vext, pmf_vext = None, None
        else:
            t1, t2, t3, freq_vext, pmf_vext = ir_args
            fn = "%s_%i_%i" %(fn, rect_idx[0], rect_idx[1])


        if self.params.plot:
            fig, axs = plt.subplots(1,3, figsize=fs)
            fig_obj = plotter.fig_obj(fig, self.nhi, self.nhj)
            axs[0] = fig_obj.phys_panel(axs[0], dat_2D, title=t1, xlabel='longitude [km]', ylabel='latitude [km]', extent=[cell.lon.min(), cell.lon.max(), cell.lat.min(), cell.lat.max()], v_extent=v_extent)

            if dfft_plot:
                axs[1] = fig_obj.fft_freq_panel(axs[1], ampls, kls[0], kls[1], typ='real', title=t2)
                axs[2] = fig_obj.fft_freq_panel(axs[2], uw, kls[0], kls[1], title=t3, typ='real')
            else:
                axs[1] = fig_obj.freq_panel(axs[1], ampls, title=t2, v_extent=freq_vext)
                axs[2] = fig_obj.freq_panel(axs[2], uw, title=t3, v_extent=pmf_vext)

            plt.tight_layout()
            if output_fig:
                plt.savefig(self.output_dir + fn + '.pdf', dpi=200, bbox_inches="tight")

            plt.show()