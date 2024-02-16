
#%%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import nibabel as nib
import seaborn as sns
from joblib import Parallel, delayed
import copy
from nilearn import image
import pandas as pd
import matplotlib.pyplot as plt
from nimare.dataset import Dataset
from nimare.decode import discrete

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

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


dset = Dataset.load('/users/local/Venkatesh/structure-function-eeg/src_data/neurosynth_dataset_filtered.pkl.gz')

res_decoded = {}
def parallelize_compute(i, n):
    b_inf = i
    b_sup = i+1/n
    filtered_brain_data = brain_data*((brain_data>np.quantile(u_values,b_inf)) & (brain_data<=np.quantile(u_values,b_sup)))
    filtered_brain_data = np.where(filtered_brain_data!=0, 1, 0)
    if filtered_brain_data.ndim == 4 and filtered_brain_data.shape[-1] == 1:
        filtered_brain_data = np.squeeze(filtered_brain_data,-1)
    

    filtered_brain = nib.Nifti1Image(filtered_brain_data,affine,dtype='int64')
    res_decoded[round(i,2)] = decode_roi(dset,filtered_brain, zscore=True).rename(columns={'r':'z_'+str(round(i,2))})
    return res_decoded


n=10
img = image.load_img('/users/local/Venkatesh/structure-function-eeg/Generated_data/rest/Graph_SDI_related/SDI_wideband_rest.nii.gz')
new_affine = copy.deepcopy(img.affine)
new_affine[0][0] = 2.0
new_affine[1][1] = 2.0
new_affine[2][2] = 2.0
U_brain = image.resample_img(img,target_affine=new_affine)
# U_brain = img
brain_data = U_brain.get_fdata()#.astype(np.float16)
brain_data[np.abs(brain_data)<1e-5] = 0
flatten_u = brain_data.flatten()
u_values = flatten_u[flatten_u!=0]
affine = U_brain.affine

res = {}
results = Parallel (n_jobs=10)(delayed(parallelize_compute)(i, n) for i in tqdm(np.arange(0,1,1/n)))

df = pd.DataFrame()
for i in range(len(results)):
    res_decoded_ = pd.concat(list(results[i].values()),axis=1)
    res_decoded_ = res_decoded_.rename(index={i:i.split('__')[1] for i in list(res_decoded_.index)})
    df[str(i*10)+'-'+str((i*10)+10)] = res_decoded_.mean(axis=1)

plt.style.use('fivethirtyeight')
def plot_heatmap(res_decoded_,min_=2.3,quantile_zero=5):
    res_decoded = res_decoded_.copy()
    
    num_col = res_decoded.shape[0]
    max_ = res_decoded.max().max()
    res_decoded[res_decoded<min_] = 0 
    heatmapOrder = getOrder(np.array(res_decoded),min_)
    sns.set(context="paper", font="sans-serif", font_scale=3)
    f, (ax1) = plt.subplots(nrows=1,ncols=1,figsize=(15, 15), sharey=True)
    plotData = res_decoded.reindex(res_decoded.index[heatmapOrder])

    cax = sns.heatmap(plotData, linewidths=2, square=False, cmap='Greys', robust=False, 
                 vmin=min_, vmax=max_, ax=ax1)
    cax.set_xticklabels(cax.get_xticklabels(), rotation=270)  # Adjust the rotation angle as needed

    cax.set_xlabel('Percentile along gradient')
    cax.set_ylabel('NeuroSynth topics terms')
    cbar = cax.collections[0].colorbar
    cbar.set_label('Zstat', rotation=270)
    cbar.set_ticks(ticks=[min_,max_])
    cbar.set_ticklabels(ticklabels=[min_,round(max_,2)])
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(0.5)

    if quantile_zero: 
        plt.vlines(quantile_zero,num_col-1,num_col,color='red')
        plt.vlines(quantile_zero+1,num_col-1,num_col,color='red')
    plt.draw()
    # f.savefig('rest_heatmap.svg', dpi=300, bbox_inches='tight')


plot_heatmap(df, 3.1, None)

# %%
