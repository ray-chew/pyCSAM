# %%
import sys
import subprocess
import os

os.chdir(os.getcwd())
print(os.getcwd())


# %%
fidx = 200
sidx = fidx+1

dd = '.'
p1 = '%s/plots_FFT_%i_%i.pdf' %(dd,fidx,sidx)
p2 = '%s/plots_FA_LSFF_%i_%i.pdf' %(dd,fidx,sidx)

out_name = '%i_%i_fft_fa_comparison' %(fidx, sidx)

ps = [p1,p2]

crop = ['pdfcrop']
for p in ps:
    cmd = crop + [p] + [p]
    proc = subprocess.call(cmd)
    print(proc)

call = ['pdfjam', '--nup', '1x2']
for p in ps:
    call += [p]

out_fn = '%s/%s.pdf' %(dd,out_name)
call += ['--outfile', out_fn]
call += ['--delta', '0.2cm 0.2cm']
print(call)

proc1 = subprocess.call(call)

proc2 = subprocess.call(crop + [out_fn] + [out_fn])

# %%

dd = '.'
p1 = '%s/plots_CSAM_%i.pdf' %(dd,fidx)
p2 = '%s/plots_CSAM_%i.pdf' %(dd,sidx)

out_name = '%i_%i_csam_plots' %(fidx, sidx)

ps = [p1,p2]

crop = ['pdfcrop']
for p in ps:
    cmd = crop + [p] + [p]
    proc = subprocess.call(cmd)
    print(proc)

call = ['pdfjam', '--nup', '1x2']
for p in ps:
    call += [p]

out_fn = '%s/%s.pdf' %(dd,out_name)
call += ['--outfile', out_fn]
call += ['--delta', '0.2cm 0.2cm']
print(call)

proc1 = subprocess.call(call)

proc2 = subprocess.call(crop + [out_fn] + [out_fn])

# %%
fidx = 42
sidx = fidx+1

dd = '.'
p1 = '%s/plots_FA_LSFF_%i_%i.pdf' %(dd, fidx, sidx)
p2 = '%s/first_plots_%i_%i.pdf' %(dd, fidx, sidx)
p3 = '%s/final_plots_%i_%i.pdf' %(dd, fidx, sidx)

out_name = '%i_%i_iter_plots' %(fidx, sidx)

ps = [p1,p2,p3]

crop = ['pdfcrop']
for p in ps:
    cmd = crop + [p] + [p]
    proc = subprocess.call(cmd)
    print(proc)

call = ['pdfjam', '--nup', '1x3']
for p in ps:
    call += [p]

out_fn = '%s/%s.pdf' %(dd,out_name)
call += ['--outfile', out_fn]
call += ['--delta', '0.2cm 0.2cm']
print(call)

proc1 = subprocess.call(call)

proc2 = subprocess.call(crop + [out_fn] + [out_fn])

# %%
dd = '.'
p1 = '%s/before_taper.pdf' %dd
p2 = '%s/mask_before_taper.pdf' %dd
p3 = '%s/after_taper.pdf' %dd
p4 = '%s/mask_after_taper.pdf' %dd

out_name = 'taper_proc'

ps = [p1,p2,p3,p4]

crop = ['pdfcrop']
for p in ps:
    cmd = crop + [p] + [p]
    print(cmd)
    proc = subprocess.call(cmd)
    print(proc)

call = ['pdfjam', '--nup', '2x2']
for p in ps:
    call += [p]

out_fn = '%s/%s.pdf' %(dd,out_name)
call += ['--outfile', out_fn]
call += ['--delta', '0.2cm 0.6cm']
print(call)

proc1 = subprocess.call(call)

proc2 = subprocess.call(crop + [out_fn] + [out_fn])
# %%
