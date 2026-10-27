import numpy as np
from glob import glob
import os

#files_list = sorted(glob("daps_enkf_safe_[0-9]*.npz"))
file_name_base = "stationary_new/test/test_"
fields_dict = {}
for i in range(0,16): 
    top_dir = file_name_base + str(i)
    file_name = os.path.join(top_dir, "daps_enkf_safe.npz")
    fields = np.load(file_name)
    for key in fields.files:
        field = fields[key]
        field = np.atleast_1d(field)
        if key not in fields_dict.keys():  
            fields_dict[key] = field
        else:
            ndims = field.ndim
            if ndims == 5:
                concat_axis = 1
            else:
                concat_axis = 0
            fields_dict[key] = np.concatenate([fields_dict[key], field], axis=concat_axis)


np.savez("stationary_new/test/daps_enkf_safe_combined_stationary.npz", **fields_dict)