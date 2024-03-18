# %%
import numpy as np

A = np.arange(1, 10).reshape(3, 3).astype(float)
print(A)

A[1, 1] = np.nan
A[2, 0] = np.nan
A[2, 2] = np.nan

print(A)
# %%
np.nanmean(A, axis=0)

# %%
A = np.arange(10)
print(A[1::3])
B = np.arange(-22, 25)
print(B[1::3])
# %%
