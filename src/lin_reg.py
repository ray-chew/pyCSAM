"""
Linear regression module
"""

import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import gmres


def get_coeffs(fobj):
    """Assembles the Fourier coefficients from the sine and cosine terms generated in the :class:`Fourier transformer class <src.fourier.f_trans>`.

    Parameters
    ----------
    fobj : :class:`src.fourier.f_trans` instance
        instance of the Fourier transformer class.

    Returns
    -------
    array-like
        2D array corresponding to the ``M`` matrix.
    """
    Ncos = fobj.bf_cos
    Nsin = fobj.bf_sin

    coeff = np.hstack([Ncos, Nsin])

    del fobj.bf_cos
    del fobj.bf_sin

    if fobj.grad:
        coeff = np.vstack([coeff, coeff])

    return coeff


def do(fobj, cell, lmbda=0.0, iter_solve=True, save_coeffs=False):
    """
    Does the linear regression

    Parameters
    ----------
    fobj : :class:`src.fourier.f_trans` instance
        instance of the Fourier transformer class.
    cell : :class:`src.var.topo_cell` instance
        cell object instance
    lmbda : float, optional
        regularisation parameter, by default 0.0
    iter_solve : bool, optional
        toggles between using direct or iterative solver, by default True
    save_coeffs : bool, optional
        skips the linear regression and just saves the generated ``M`` matrix for diagnostics and debugging, by default False

    Returns
    -------
    a_m : list
        list of Fourier amplitudes corresponding to the unknown vector in the linear problem
    data_recons : like
        vector-like topography reconstructed from ``a_m``
    """
    if fobj.grad:
        cell.get_grad()
        data = cell.grad_topo_m
    else:
        data = cell.topo_m

    coeff = get_coeffs(fobj)

    if save_coeffs:
        fobj.coeff = coeff
        return None, None

    # tot_coeff = coeff.shape[1]

    # E_tilda_lm = np.zeros((tot_coeff,tot_coeff))

    h_tilda_l = np.dot(coeff.T, data.reshape(-1, 1)).flatten()

    E_tilda_lm = np.dot(coeff.T, coeff)

    trace = np.trace(E_tilda_lm) / len(np.diag(E_tilda_lm)) * lmbda
    szc = E_tilda_lm.shape[0]
    for ttr in range(szc):
        E_tilda_lm[ttr, ttr] += trace

    if iter_solve:
        a_m, _ = gmres(E_tilda_lm, h_tilda_l)
    else:
        a_m = la.inv(E_tilda_lm).dot(h_tilda_l)

    # regular FFT considers normalization by total nu  mber of datapoints N=100
    # so multiply the Fourier coefficients by N here
    # a_m = a_m#*len(data)

    data_recons = coeff.dot(a_m)

    return a_m, data_recons
