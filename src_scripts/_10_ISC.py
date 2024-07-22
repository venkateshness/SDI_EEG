#%%
import util_5_CorrCA
import numpy as np

HOMEDIR = "/users/local/Venkatesh/structure-function-eeg/"
fs = 125

ISC_bundle = dict()
for band in ["theta", 'widerband']:
    if band == "theta":
        envelope = np.array(list(np.load(f"{HOMEDIR}/Generated_data/video1/cortical_surface_related/{band}_bandpassed.npz").values()))
    elif band=="widerband":
        envelope = np.array(list(np.load(f"{HOMEDIR}/Generated_data/video1/cortical_surface_related/parcellated_widerband.npz").values()))
    dic = dict()
    dic["condition1"] = envelope
    W, _ = util_5_CorrCA.train_cca(dic)
    ISC_bundle[f'{band}'] = util_5_CorrCA.apply_cca(dic["condition1"], W, fs)[1]

np.savez_compressed(f"{HOMEDIR}/Generated_data/video1/cortical_surface_related/ISC_bundle.npz", **ISC_bundle)
