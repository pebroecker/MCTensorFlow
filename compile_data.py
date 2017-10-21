import h5py
from glob import glob
from shutil import copyfile
import os
import sys

in_pattern = "**/honeycomb_L_6_W_6_theta_20_dt_0.1_*.task{t}.out.h5"
out_pattern = "honeycomb_L_6_W_6_theta_20_dt_0.1.task{t}.out.h5"
to_copy = ["greens_imag", "greens_real"]

for t in range(100):
    files = glob(in_pattern.format(t=t, recursive=True))
    if len(files) == 0:
        continue
        
    out_filename = out_pattern.format(t=t)
    out_file = h5py.File(out_filename, "w")
    
    id = 1
    for i,f in enumerate(files[1:]):
        in_file = h5py.File(f, "r")
        if i == 1:
            in_file.copy("parameters", out_file)

        try:
            for tc in to_copy:
                
                out_file["simulation/results/{}/{}".format(tc, id)] = in_file["simulation/results/{}/1".format(tc)][...].reshape(128, 72, 72)
                out_file["simulation/results/{}/{}".format(tc, id + 1)] = in_file["simulation/results/{}/2".format(tc)][...].reshape(128, 72, 72)
            in_file.close()
            id += 2
        except:
            pass
            
    for f in files:
        print("Deleting ", f)
        os.remove(f)
