import numpy as np
from src import physics
from vis import plotter
from copy import deepcopy

import matplotlib.pyplot as plt

class delaunay_metrics(object):

    def __init__(self, params, tri, writer=None):
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
        self.uw_ref = uw_ref.sum()
        self.uw_fa = uw_fa.sum()

        self.idx_name.append(idx)
        self.pmf_refs.append(self.uw_ref)
        self.pmf_fas.append(self.uw_fa)


    def get_rel_err(self, triangle_pair):
        self.update_pair(triangle_pair, store_error=False)
        self.rel_err = self.get_rel_diff(self.uw_sum, self.uw_ref)

        return self.rel_err


    def update_pair(self, triangle_pair, store_error=True):
        for triangle in triangle_pair:
            assert hasattr(triangle, 'analysis'), "triangle has no analysis object."

        self.t0 = triangle_pair[0]
        self.t1 = triangle_pair[1]

        self.uw_sum = self.get_pmf_sum()
        self.uw_spec_sum = self.get_pmf_spec_sum()

        if store_error:
            self.pmf_sums.append(self.uw_sum)
            self.pmf_ssums.append(self.uw_spec_sum)


    def get_pmf_sum(self):
        self.uw_0 = self.t0.uw.sum()
        self.uw_1 = self.t1.uw.sum()

        return self.uw_0 + self.uw_1
        

    def get_pmf_spec_sum(self):
        self.ampls_0 = self.t0.analysis.ampls
        self.ampls_1 = self.t1.analysis.ampls
        self.ampls_sum = (self.ampls_0 + self.ampls_1)

        # consider replacing deepcopy with copy method.
        analysis_sum = deepcopy(self.t0.analysis)
        analysis_sum.ampls = self.ampls_sum

        ideal = physics.ideal_pmf(U=self.params.U, V=self.params.V)

        return 0.5 * ideal.compute_uw_pmf(analysis_sum)


    def __repr__(self):
        errs = [self.uw_ref, self.uw_fa, self.uw_sum, self.uw_spec_sum]
        errs = [str(err) for err in errs]
        uw_strs = str(self.uw_0) + ', ' +  str(self.uw_1)
        err_strs = ', '.join(errs)

        return uw_strs + '\n' + err_strs + '\n'
    
    def __str__(self):
        return repr(self)
    

    def end(self, verbose=False):
        self.gen_percentage_errs()
        self.gen_regional_errs()

        if self.writer is not None:
            self.write()

        if verbose:
            print(np.abs(self.max_errs).mean(), np.abs(self.rel_errs).mean())


    def write(self):
        assert self.writer is not None

        self.writer.populate('decomposition', 'pmf_refs', self.pmf_refs)
        self.writer.populate('decomposition', 'pmf_fas', self.pmf_fas)
        self.writer.populate('decomposition', 'pmf_sums', self.pmf_sums)
        self.writer.populate('decomposition', 'pmf_ssums', self.pmf_ssums)

    def gen_percentage_errs(self):
        max_idx = np.argmax(np.abs(self.pmf_refs))
        self.max_errs = self.get_max_diff(self.pmf_sums, self.pmf_refs, np.array(self.pmf_refs[max_idx]))
        self.rel_errs = self.get_rel_diff(self.pmf_sums, self.pmf_refs)

        self.max_errs = np.array(self.max_errs) * 100
        self.rel_errs = np.array(self.rel_errs) * 100

    def gen_regional_errs(self):
        assert hasattr(self, 'max_errs')
        assert hasattr(self, 'rel_errs')

        self.reg_max_errs = self.get_regional_errs(self.tri, self.max_errs)
        self.reg_rel_errs = self.get_regional_errs(self.tri, self.rel_errs)

    def get_regional_errs(self, tri, err):
        errors = np.zeros((len(tri.simplices)))
        errors[:] = np.nan
        errors[self.params.rect_set] = err
        errors[np.array(self.params.rect_set)+1] = err

        return errors

    @staticmethod
    def get_rel_diff(arr, ref):
        arr = np.array(arr)
        ref = np.array(ref)

        return arr / ref - 1.0

    @staticmethod
    def get_max_diff(arr, ref, max):
        arr = np.array(arr)
        ref = np.array(ref)

        return (arr - ref) / max
    

class diag_plotter(object):

    def __init__(self, params, nhi, nhj):
        self.params = params
        self.nhi = nhi
        self.nhj = nhj

        self.output_dir = "../manuscript/"

    def show(self, rect_idx, sols, kls=None, v_extent=None, dfft_plot=False, output_fig=True, fs = (14.0,4.0), ir_args=None, fn=None):

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