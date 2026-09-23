import numpy as np
from glob import glob

files_list = sorted(glob("daps_enkf_safe_[0-9]*.npz"))

fields_dict = {}
for file_name in files_list[1:4]:
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
            print(fields_dict[key].shape)

print(fields_dict["posterior"].shape)

np.savez("daps_enkf_safe_combined_75_first_25_removed.npz", **fields_dict)