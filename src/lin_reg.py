import numpy as np
import scipy.linalg as la

def do(fobj, cell):
    Ncos = fobj.bf_cos
    Nsin = fobj.bf_sin

    nhar_i = fobj.nhar_i
    nhar_j = fobj.nhar_j

    data = cell.topo_m

    coeff = np.hstack([Ncos,Nsin])
    tot_coeff = coeff.shape[1]

    E_tilda_lm = np.zeros((tot_coeff,tot_coeff))

    h_tilda_l = np.dot(coeff.T, data.reshape(-1,1)).flatten()

    E_tilda_lm = np.dot(coeff.T, coeff)

    a_m = la.inv(E_tilda_lm).dot(h_tilda_l)

    # regular FFT considers normalization by total nu  mber of datapoints N=100
    # so multiply the Fourier coefficients by N here
    # a_m = a_m#*len(data)

    data_recons = coeff.dot(a_m)

    return a_m, data_recons