# %%
import sys
import os
# set system path to find local modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

from src import var, utils
from wrappers import interface
from vis import plotter
from copy import deepcopy

from IPython import get_ipython
ipython = get_ipython()

if ipython is not None:
    ipython.run_line_magic('load_ext', 'autoreload')

def autoreload():
    if ipython is not None:
        ipython.run_line_magic('autoreload', '2')

autoreload()

# %%
#### generate random values for the artificial terrain
np.random.seed(777)

sz = 25
nk = np.random.randint(0,12, size=sz)
nl = np.random.randint(-5,7, size=sz)

for ii  in range(sz):
    if nk[ii] == 0 and nl[ii] < 0:
        nk[ii] += np.random.randint(1,11)
pts = [item for item in zip(nk,nl)]

pts = np.array(list(set(pts)))

nk = pts[:,0]
nl = pts[:,1]

sz = len(pts)

Ak = np.random.random(size=sz) * 100.0
Al = np.random.random(size=sz) * 100.0

sck = np.random.randint(0,2,size=sz)
scl = np.random.randint(0,2,size=sz)

nhi = 12
nhj = 12
freqs_ref = np.zeros((nhi,nhj))

cnt = 0
for pt in pts:
    kk, ll = pt
    ll += 5
    print(kk,ll)
    freqs_ref[ll, kk] = Ak[cnt]
            
    cnt += 1

print("number of unique modes:", sz)
ref_sum = freqs_ref.sum()

# %%

#### run parameters
n_modes = 14
lmbda_reg = 8.0*1e-5
lmbda_fg = 1e-1
lmbda_sg = 1e-6

#### define wavenumber range
nhi = 12
nhj = 12

ll = np.arange(-(nhj/2-1),nhj/2+1)
kk = np.arange(0,nhi)

#### initialise triangle
grid = var.grid()
cell = var.topo_cell()

vid = utils.isosceles(grid, cell)

lat_v = grid.clat_vertices[vid,:]
lon_v = grid.clon_vertices[vid,:]

cell.gen_mgrids()

#### fill artificial topography
cell.topo = np.cos(1.0 * cell.lat_grid) + np.sin(5.0 * cell.lon_grid)
cell.topo[...] = 0.0

def sinusoidal_basis(Ak, nk, Al, nl, sc, typ):
    nk = 2.0 * np.pi * nk / cell.lon.max()
    nl = 2.0 * np.pi * nl / cell.lat.max()

    if sc == 0:
        bf = Ak * np.cos(nk * cell.lon_grid + nl * cell.lat_grid)
    else:
        bf = Al * np.sin(nk * cell.lon_grid + nl * cell.lat_grid)
    
    return bf

for ii in range(sz):
    cell.topo += sinusoidal_basis(Ak[ii], nk[ii], Al[ii], nl[ii], sck[ii], 'k')

#### define triangle given the vertices
triangle = utils.gen_triangle(lon_v, lat_v)
cell.get_masked(triangle=triangle)

cell.wlat = np.diff(cell.lat).mean()
cell.wlon = np.diff(cell.lon).mean()

#### define quadrilateral counterpart
cell_quad = deepcopy(cell)
cell_quad.get_masked(mask=np.ones_like(cell.topo).astype('bool'))

#### artificial winds, we do not need them in the idealised test
U, V = 1.0, 1.0

pure_lsff = interface.get_pmf(nhi,nhj,U,V)
reg_lsff = interface.get_pmf(nhi,nhj,U,V)

#### number of experiments we're running + 1 for the reference run
num_experiments = 6 

freqs_arr = np.zeros((num_experiments, nhi, nhj))
dat_arr = np.array([None]*num_experiments, dtype=object)

#### helper function to run the CSAM algorithm
def csam_run(cell, n_modes, lmbda_fg, lmbda_sg):
    first_guess = interface.get_pmf(nhi,nhj,U,V)

    cell.get_masked(mask=np.ones_like(cell.topo).astype('bool'))

    cell.wlat = np.diff(cell.lat).mean()
    cell.wlon = np.diff(cell.lon).mean()

    freqs_fg, _, dat_2D_fg = first_guess.sappx(cell, lmbda=lmbda_fg, iter_solve=False)

    fq_cpy = np.copy(freqs_fg)
    fq_cpy[np.isnan(fq_cpy)] = 0.0 # necessary. Otherwise, popping with fq_cpy.max() gives the np.nan entries first.

    indices = []
    max_ampls = []

    for ii in range(n_modes):
        max_idx = np.unravel_index(fq_cpy.argmax(), fq_cpy.shape)
        indices.append(max_idx)
        max_ampls.append(fq_cpy[max_idx])
        max_val = fq_cpy[max_idx]
        fq_cpy[max_idx] = 0.0

    k_idxs = [pair[1] for pair in indices]
    l_idxs = [pair[0] for pair in indices]

    second_guess = interface.get_pmf(nhi,nhj,U,V)

    second_guess.fobj.set_kls(k_idxs, l_idxs, recompute_nhij = False)

    cell.get_masked(triangle=triangle)

    cell.wlat = np.diff(cell.lat).mean()
    cell.wlon = np.diff(cell.lon).mean()

    freqs, _, dat_2D = second_guess.sappx(cell, lmbda=lmbda_sg, updt_analysis=True, scale=1.0, iter_solve=False)

    return freqs, _, dat_2D

# %%
#### reference run
freqs_arr[0], dat_arr[0] = freqs_ref, cell.topo * cell.mask

#### pure lsff run
freqs_arr[1], _, dat_arr[1] = pure_lsff.sappx(cell, lmbda=0.0, iter_solve=False, save_am=True)

#### pure lsff run recomputed on the full quadrilateral domain
freqs_arr[5], _, dat_arr[5] = pure_lsff.recompute_rhs(cell_quad, pure_lsff.fobj, save_coeffs=True)

#### regularised lsff run
freqs_arr[2], _, dat_arr[2] = reg_lsff.sappx(cell, lmbda=lmbda_reg, iter_solve=False)

#### optimal CSAM run
freqs_arr[3], _, dat_arr[3] = csam_run(cell, sz, lmbda_fg, lmbda_sg)

#### suboptimal CSAM run
freqs_arr[4], _, dat_arr[4] = csam_run(cell, n_modes, lmbda_fg, lmbda_sg)

freqs_arr = np.array([np.nan_to_num(freq) for freq in freqs_arr])

print(freqs_arr.shape)

errs = np.array([np.linalg.norm(freq - freqs_ref) for freq in freqs_arr])
sums = np.array([freq.sum() for freq in freqs_arr])
sum_errs = np.array([np.abs(freq.sum() - freqs_arr[0].sum()) / freqs_arr[0].sum() for freq in freqs_arr])



# %%
autoreload()
#### plot the idealised runs

#### which results do we want to plot?
idxs = [0,2,3,4]

fs = (10,4.5)
fig, axs = plt.subplots(2,len(idxs), figsize=fs, sharey='row')
fig_obj = plotter.fig_obj(fig, pure_lsff.fobj.nhar_i, pure_lsff.fobj.nhar_j, cbar=False, set_label=False)

selected_errs = []
selected_sums = []
selected_sum_errs = []

phys_lbls = ["reference", "pLSFF", "optCSAM", "subCSAM"]
spec_lbls = ["", "", "", ""]

for cnt, idx in enumerate(idxs):
    freq = freqs_arr[idx]
    dat = dat_arr[idx]

    axs[0,cnt] = fig_obj.phys_panel(axs[0,cnt], dat, title=phys_lbls[cnt], v_extent=[dat_arr[0].min(),dat_arr[0].max()])
    axs[1,cnt] = fig_obj.freq_panel(axs[1,cnt], freq, title=spec_lbls[cnt], v_extent=[freqs_arr[0].min(), freqs_arr[0].max()])

    if cnt > 0:
        selected_errs.append(errs[idx])
        selected_sum_errs.append(sum_errs[idx])
    selected_sums.append(sums[idx])

fig.colorbar(axs[0,-1].get_images()[0], ax=axs[0, :], fraction=0.046, pad=0.04)
fig.colorbar(axs[1,-1].get_children()[0], ax=axs[1, :], fraction=0.046, pad=0.04)

axs[1,0].set_ylabel("$m$", fontsize=12)

# plt.tight_layout()
plt.savefig('../manuscript/idealized_plots.pdf', bbox_inches="tight")
plt.show()


# %%
#### plot the errors
print(sums)
plotter.error_bar_abs_plot(selected_errs, phys_lbls[1:], color=['C0', 'C1','C2'], ylims=[0,140], title=r'$L_2$-error in the spectrum', fontsize=14, fs=(3.5,2.5), output_fig=True, fn='../manuscript/l2_errs.pdf')
plotter.error_bar_abs_plot(selected_sums, phys_lbls, color=['C3','C0','C1','C2'], ylims=[0,2200], title='total abs. amplitude', fontsize=14, fs=(4.5,2.5), output_fig=True, fn='../manuscript/powers.pdf')


# %%
#### print the errors
np.set_printoptions(suppress=True)
print(np.around(sum_errs,5)*100)
print(errs)

# %%
#### plot the overfitting issue
fs = (10.0,2.8)
fig, axs = plt.subplots(1,3, figsize=fs, gridspec_kw={'width_ratios': [1, 1, 1]})
fig_obj = plotter.fig_obj(fig, pure_lsff.fobj.nhar_i, pure_lsff.fobj.nhar_j, cbar=False, set_label=False)

selected_errs = []
selected_sums = []
selected_sum_errs = []

phys_lbls = ["non-quad. reconst.", "quadrilateral reconst."]
spec_lbls = ["power spectrum",]

for cnt, idx in enumerate([1,5]):
    freq = freqs_arr[idx]
    dat = dat_arr[idx]

    axs[cnt] = fig_obj.phys_panel(axs[cnt], dat, title=phys_lbls[cnt], v_extent=[dat_arr[0].min(),dat_arr[0].max()])

    fig.colorbar(axs[cnt].get_images()[0], ax=axs[cnt])

    if cnt == 0:
        axs[2] = fig_obj.freq_panel(axs[2], freq, title=spec_lbls[cnt], v_extent=[freqs_arr[cnt].min(), freqs_arr[cnt].max()])

        fig.colorbar(axs[2].get_children()[0], ax=axs[2])

axs[2].set_ylabel("$m$", fontsize=12)

plt.tight_layout()
plt.savefig('../manuscript/overfitting_issue.pdf', bbox_inches="tight")
plt.show()

# %%
