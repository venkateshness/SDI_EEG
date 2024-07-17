#%%
import numpy as np
from scipy import io as sio
import matplotlib.pyplot as plt
from os.path import join as opj
from tqdm import tqdm

import pickle
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import seaborn as sns
from joblib import Parallel, delayed
from nilearn import image
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from nimare.dataset import Dataset
from nimare.decode import discrete,continuous
from nimare.utils import get_resource_path
from nimare.decode.discrete import NeurosynthDecoder

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
HOMEDIR = "/users/local/Venkatesh/structure-function-eeg/"

def decode_roi(dataset,img,zscore=False):
    decoder = discrete.ROIAssociationDecoder(img)
    decoder.fit(dataset)#dset.slice(m)
    decoded_df_0 = decoder.transform()
    if zscore :
        decoded_df_0.r = decoded_df_0.r.apply(lambda r: np.arctanh(r) * np.sqrt(len(dataset.annotations) - 3))
    print(decoded_df_0)
    return decoded_df_0.sort_values('r')

def getOrder(d, thr):
    dh = []
    for i in range(0,len(d)):
        di = d[i]
        try : 
            dh.append(np.average(np.array(range(0,len(d[i]))) + 1, weights=di))
        except :
            dh.append(0)
    heatmapOrder = np.argsort(dh)
    return heatmapOrder

dataset = Dataset.load(f'{HOMEDIR}/src_data/neurosynth_dataset_filtered.pkl.gz')
 
def parallelize_compute(i, nifti):
        
    res_decoded[round(i,2)] = decode_roi(dataset, f'{HOMEDIR}/decoding_images/{condition}/{nifti}', zscore=True)
    decoded = pd.concat(list(res_decoded.values()),axis=1)
    res_decoded = res_decoded.rename(index={i:i.split('__')[1] for i in list(res_decoded.index)})
    
    return res_decoded
condition = 'video1'
results = Parallel (n_jobs=5, backend='loky')(delayed(parallelize_compute)(i, nifti) for i, nifti in tqdm(enumerate(sorted(os.listdir(f'{HOMEDIR}/decoding_images/{condition}')))))

# %%
decode_roi(dataset, f'/users/local/Venkatesh/structure-function-eeg/decoding_images/video1/SDI_video1_0.0.nii.gz', zscore=True)

# %%
