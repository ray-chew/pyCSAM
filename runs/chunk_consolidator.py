# %%
import numpy as np
from tqdm import tqdm

from pycsam.src import io, var
from pycsam.inputs.icon_global_run import params

chunk_start = 0
n_cells     = 20480
chunk_sz    = 100

dat_path = params.path_output + "global_dataset/chunks/"
out_path = params.path_output + "global_dataset/"
out_fn = 'icon_global_R2B4'

global_dat = np.zeros((n_cells), dtype='object')

cnt = 0
for chunk in tqdm(range(chunk_start, n_cells, chunk_sz)):

    sfx = "_" + str(chunk+chunk_sz)
    fn = params.fn_output + sfx + '.nc'

    writer = io.nc_writer(params, sfx)

    if chunk+chunk_sz > n_cells:
        chunk_end = n_cells
    else:
        chunk_end = chunk+chunk_sz

    for ii in range(chunk, chunk_end):
        struct = var.obj()
        res = writer.read_dat(dat_path, fn, ii, struct)
        global_dat[cnt] = struct
        # print(cnt)
        del struct

        cnt += 1

# print(cnt, chunk_end)
print("\n==========")
print("Collection done; writing output...")
print("==========\n")
assert (cnt) == chunk_end

params.path_output = out_path
global_writer = io.nc_writer(params, '')

for cnt, item in tqdm(enumerate(global_dat)):
    global_writer.duplicate(cnt, item)

# %%
