# import required module
import os
import secrets

import numpy as np

import utils
import shutil

# assign directory
directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results")
merged_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "merged-results")

shutil.rmtree(merged_dir) # remove dir and all content (clean-up)
os.makedirs(merged_dir, exist_ok=True)

unique_params = []
merged_data = []

for filename in os.listdir(directory):
    f = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results", filename)
    data, params = utils.data_from_file(f)
    param_unique = True
    ip = None
    print(params)
    for ip, p in enumerate(unique_params):
        if utils.is_same_params(params, p):
            param_unique = False
            break
    if param_unique:
        print(f"New dataset found {filename}")
        unique_params.append(params)
        merged_data.append(data)
    else:
        # merging time
        print(f"Merging dataset {filename}")
        res = utils.merge(merged_data[ip], data)
        if res is not None:
            merged_data[ip] = res
    print("----------------------------------------------")

# write to file
for params, data in zip(unique_params, merged_data):
    np.savez_compressed(
        os.path.join(merged_dir, f"data-merged.npz"),
        pa_prior_csi=data["pa_prior_csi"],
        md_prior_csi=data["md_prior_csi"], pa_partial_csi_ZF=data["pa_partial_csi_ZF"],
        md_partial_csi_ZF=data["md_partial_csi_ZF"],
        pa_partial_csi=data["pa_partial_csi"], md_partial_csi=data["md_partial_csi"], params=params, SHAPE_PROB=data["SHAPE_PROB"])
