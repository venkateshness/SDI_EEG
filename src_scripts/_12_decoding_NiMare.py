
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
import numpy as np
import nibabel as nib
import os
HOMEDIR = "/users/local/Venkatesh/structure-function-eeg/"
from nilearn.regions import signals_to_img_labels
from nilearn.datasets import fetch_icbm152_2009
mnitemp = fetch_icbm152_2009()
path_Glasser = f"{HOMEDIR}/src_data/Glasser_masker.nii.gz"

#%%
HOMEDIR = "/users/local/Venkatesh/structure-function-eeg/"
from nilearn.regions import signals_to_img_labels
from nilearn.datasets import fetch_icbm152_2009
mnitemp = fetch_icbm152_2009()
path_Glasser = f"{HOMEDIR}/src_data/Glasser_masker.nii.gz"
import numpy as np
from nilearn import image

n=10
condition = 'rest'
empi_SDI = np.log2(np.mean(np.squeeze(np.load(HOMEDIR + f"Generated_data/{condition}/Graph_SDI_related/empirical_SDI.npz")[f'widerband']), axis=0))

for i in np.arange(0,1,1/n):
    b_inf = i
    b_sup = i+1/n
    
    lower = np.quantile(empi_SDI, b_inf)
    upper = np.quantile(empi_SDI, b_sup)

    binarized = np.where(np.logical_and(empi_SDI>lower, empi_SDI<=upper))
    
    zeros = np.zeros(empi_SDI.shape)
    zeros[binarized[0]] = 1
    
    nifti= signals_to_img_labels(zeros, path_Glasser, mnitemp["mask"])
    nifti.to_filename(f'{HOMEDIR}/Generated_data/decoding_images/{condition}/SDI_{condition}_{round(i,2)}.nii.gz')

# %%


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def decode_roi(dataset,img,zscore=False):
    decoder = discrete.ROIAssociationDecoder(img)
    decoder.fit(dataset)#dset.slice(m)
    decoded_df_0 = decoder.transform()
    if zscore :
        decoded_df_0.r = decoded_df_0.r.apply(lambda r: np.arctanh(r) * np.sqrt(len(dataset.annotations) - 3))

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


dset = Dataset.load(f'{HOMEDIR}/src_data/neurosynth_dataset_filtered.pkl.gz')

res_decoded = {}
def parallelize_compute(i, nii):
    nifti = nib.load(f'{HOMEDIR}/Generated_data/decoding_images/{condition}/{nii}')
    res_decoded[round(i,2)] = decode_roi(dset, nifti, zscore=True).rename(columns={'r':'z_'+str(round(i,2))})
    return res_decoded

n=10
condition = 'video1'

res = {}
results = Parallel (n_jobs=5)(delayed(parallelize_compute)(i, nii) for i, nii in tqdm(enumerate(sorted(os.listdir(f'{HOMEDIR}/Generated_data/decoding_images/{condition}')))))
#%%


df = pd.DataFrame()
for i in range(len(results)):
    res_decoded_ = pd.concat(list(results[i].values()),axis=1)
    res_decoded_ = res_decoded_.rename(index={i:i.split('__')[1] for i in list(res_decoded_.index)})
    df[str(i*10)+'-'+str((i*10)+10)] = res_decoded_.mean(axis=1)

plt.style.use('fivethirtyeight')


index = (['42_visual_cortex_sensory', '17_motor_cortex_hand', '6_auditory_speech_temporal',        '32_pain_somatosensory_stimulation', '45_motion_perception_visual', 

'40_face_faces_facial', '9_memory_working_wm', '16_response_inhibition_control',
'37_language_reading_word', '47_attention_attentional_target', '18_number_ips_numerical',       '19_action_actions_observation', 

'7_reward_feedback_striatum', '20_control_conflict_task', 
 '30_decision_making_risk', '28_social_empathy_moral', '26_emotional_amygdala_negative',  '41_imagery_mental_events', 
       '33_memory_retrieval_encoding', 
       '8_mpfc_social_medial'])



plt.style.use('fivethirtyeight')
def plot_heatmap(min_=2.3, plotData=None):
    
    max_ = 12.45
    
    sns.set(context="paper", font= 'sans-serif', font_scale=5)
    f, (ax1) = plt.subplots(nrows=1,ncols=1,figsize=(25, 25), sharey=True)
    # plotData = res_decoded.reindex(index)

    
    
    cax = sns.heatmap(plotData, linewidths=1, square=False, cmap='RdPu', robust=False, 
            ax=ax1, vmin=min_, vmax=max_, mask=plotData == 0)
    cax.set_xticklabels(cax.get_xticklabels(), rotation=270)  # Adjust the rotation angle as needed

    cax.set_xlabel('Percentile along Coupling-Decoupling gradient')
    cax.set_ylabel('NeuroSynth topics terms')
    cbar = cax.collections[0].colorbar
    cbar.set_label('Zstat', rotation=270)
    cbar.set_ticks(ticks=[min_,max_])
    cbar.set_ticklabels(ticklabels=[min_,round(max_,2)])
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(0.5)

    
    plt.draw()
    f.savefig(f'{HOMEDIR}/Results/Figure_5/{condition}_heatmap.svg', dpi=300, bbox_inches='tight')
    # np.savez_compressed(f'{HOMEDIR}/Generated_data/{condition}/decoding_{condition}_heatmap.npz', data=plotData)
    



# plot_heatmap(df, 3.1, None)

# %%
condition = 'rest'

signal = pd.DataFrame(np.load(f'{HOMEDIR}/Generated_data/{condition}/decoding_{condition}_heatmap.npz')['data'])
signal.index =  (['Visual_cortex_sensory', 'Motor_cortex_hand', 'Auditory_speech_temporal', 'Pain_somatosensory_stimulation', 'Motion_perception_visual', 

'Face_faces_facial', 'Memory_working_wm', 'Response_inhibition_control',
'Language_reading_word', 'Attention_attentional_target', 'Number_ips_numerical', 'Action_actions_observation', 

'Reward_feedback_striatum', 'Control_conflict_task', 
 'Decision_making_risk', 'Social_empathy_moral', 'Emotional_amygdala_negative',  'Imagery_mental_events', 
       'Memory_retrieval_encoding', 
       'Mpfc_social_medial'])
signal.columns = [str(i*10)+'-'+str((i*10)+10) for i in range(10)]

plot_heatmap(min_=3.1, plotData=signal)
# %%

from nilearn import plotting
condition = 'video1'

nifti = nib.load(f'{HOMEDIR}/Generated_data/decoding_images/{condition}/SDI_{condition}_0.9.nii.gz')
plotting.plot_img_on_surf(stat_map=nifti, title='SDI', cmap='seismic', colorbar=True, symmetric_cbar="auto")
# %%
