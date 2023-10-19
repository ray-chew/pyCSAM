import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import gmres

def get_coeffs(fobj):
    Ncos = fobj.bf_cos
    Nsin = fobj.bf_sin

    coeff = np.hstack([Ncos,Nsin])

    del fobj.bf_cos
    del fobj.bf_sin

    if fobj.grad: coeff = np.vstack([coeff,coeff])

    return coeff


def do(fobj, cell, lmbda = 0.0):
    if fobj.grad:
        cell.get_grad()
        data = cell.grad_topo_m
    else:
        data = cell.topo_m

    coeff = get_coeffs(fobj)

    # tot_coeff = coeff.shape[1]

    # E_tilda_lm = np.zeros((tot_coeff,tot_coeff))

    h_tilda_l = np.dot(coeff.T, data.reshape(-1,1)).flatten()

    E_tilda_lm = np.dot(coeff.T, coeff)

    trace = np.trace(E_tilda_lm) / len(np.diag(E_tilda_lm)) * lmbda
    szc = E_tilda_lm.shape[0]
    for ttr in range(szc):
        E_tilda_lm[ttr,ttr] += trace 

    a_m = la.inv(E_tilda_lm).dot(h_tilda_l)

    # regular FFT considers normalization by total nu  mber of datapoints N=100
    # so multiply the Fourier coefficients by N here
    # a_m = a_m#*len(data)

    data_recons = coeff.dot(a_m)

    return a_m, data_recons


def do_iter(fobj, cell, lmbda = 0.0, get_n_modes=False):
    if fobj.grad:
        cell.get_grad()
        data = cell.grad_topo_m
    else:
        data = cell.topo_m

    coeff = get_coeffs(fobj)

    # tot_coeff = coeff.shape[1]

    # E_tilda_lm = np.zeros((tot_coeff,tot_coeff))

    h_tilda_l = np.dot(coeff.T, data.reshape(-1,1)).flatten()

    E_tilda_lm = np.dot(coeff.T, coeff)

    trace = np.trace(E_tilda_lm) / len(np.diag(E_tilda_lm)) * lmbda
    szc = E_tilda_lm.shape[0]
    for ttr in range(szc):
        E_tilda_lm[ttr,ttr] += trace 

    a_m, _ = gmres(E_tilda_lm, h_tilda_l)

    if get_n_modes:
        # to be replaced with params.n_modes
        a_m[np.where(a_m < np.sort(a_m.ravel())[::-1][99])] = 0.0

    print( (a_m > 0.0).sum() )

    # regular FFT considers normalization by total nu  mber of datapoints N=100
    # so multiply the Fourier coefficients by N here
    # a_m = a_m#*len(data)

    data_recons = coeff.dot(a_m)

    return a_m, data_recons