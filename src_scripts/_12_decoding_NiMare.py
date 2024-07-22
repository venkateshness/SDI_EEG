# Credits to Yassine El Ouahidi for sharing us the NiMare code for decoding, originally applied using Neurosynth 

from tqdm import tqdm
import seaborn as sns
from joblib import Parallel, delayed
import copy
from nilearn import image
import pandas as pd
import matplotlib.pyplot as plt
from nimare.dataset import Dataset
from nimare.decode import discrete
import numpy as np
from nilearn import plotting
import nibabel as nib
import os
from nilearn.regions import signals_to_img_labels
from nilearn.datasets import fetch_icbm152_2009
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

mnitemp = fetch_icbm152_2009()
HOMEDIR = "/users/local/Venkatesh/structure-function-eeg/"
path_Glasser = f"{HOMEDIR}/src_data/Glasser_masker.nii.gz"

n=10
condition = 'video2' 
empi_SDI = np.log2(np.mean(np.squeeze(np.load(HOMEDIR + f"Generated_data/{condition}/Graph_SDI_related/empirical_SDI.npz")[f'widerband']), axis=0))

# export the unthresholded spatial into percentile bins
for i in np.arange(0,1,1/n):
    b_inf = i
    b_sup = i+1/n
    
    lower = np.quantile(empi_SDI, b_inf)
    upper = np.quantile(empi_SDI, b_sup)

    binarized = np.where(np.logical_and(empi_SDI>lower, empi_SDI<=upper))
    
    zeros = np.zeros(empi_SDI.shape)
    zeros[binarized[0]] = 1
    
    nifti= signals_to_img_labels(zeros, path_Glasser, mnitemp["mask"])
    nifti.to_filename(f'{HOMEDIR}/decoding_images/{condition}/SDI_{condition}_{round(i,2)}.nii.gz')



def decode_roi(dataset, img, zscore=False):
    """Perform decoding using a ROI association decoder.

    Args:
        dataset: Neurosynth dataset
        img : Nifti image
        zscore (bool, optional): Zscoring. Defaults to False.

    Returns:
        dataframe: decoded data in dataframe
    """
    
    decoder = discrete.ROIAssociationDecoder(img)
    decoder.fit(dataset)
    decoded_df_0 = decoder.transform()
    if zscore :
        decoded_df_0.r = decoded_df_0.r.apply(lambda r: np.arctanh(r) * np.sqrt(len(dataset.annotations) - 3))
    return decoded_df_0.sort_values('r')

res_decoded = {}
def parallelize_compute(i, nii):
    """Parallelize the decoding process

    Args:
        i : position of the nii file
        nii : Nifti files

    Returns:
        dataframe: decoded data in dataframe
    """
    nifti = nib.load(f'{HOMEDIR}/decoding_images/{condition}/{nii}')
    
    new_affine = copy.deepcopy(nifti.affine)
    new_affine[0][0] = 2.0
    new_affine[1][1] = 2.0
    new_affine[2][2] = 2.0
    
    U_brain = image.resample_img(nifti,target_affine=new_affine)
    data = U_brain.get_fdata()
    filtered_brain = nib.Nifti1Image(data.astype(np.int_),new_affine,dtype='int64')

    res_decoded[round(i,2)] = decode_roi(dset, filtered_brain, zscore=True).rename(columns={'r':'z_'+str(round(i,2))})
    return res_decoded

res = {}
results = Parallel (n_jobs=10, backend='loky')(delayed(parallelize_compute)(i, nii) for i, nii in tqdm(enumerate(sorted(os.listdir(f'{HOMEDIR}/decoding_images/{condition}')))))
dset = Dataset.load(f'{HOMEDIR}/src_data/neurosynth_dataset_filtered.pkl.gz')

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
       '8_mpfc_social_medial']) # Manual ordering; rationale explained in the Section 2.4 (Decoding of the SDI maps)


plt.style.use('fivethirtyeight')
def plot_heatmap(min_=2.3):
    """Plot the decoded results in heatmap 

    Args:
        min_ (float, optional): Min value for the heatmap. Defaults to 2.3.
    """
    
    max_ = 12.45
    
    sns.set(context="paper", font= 'sans-serif', font_scale=5)
    f, (ax1) = plt.subplots(nrows=1,ncols=1,figsize=(25, 25), sharey=True)
    plotData = df.reindex(index)    
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
    # np.savez_compressed(f'{HOMEDIR}/revision/Data_for_plots/decoding_{condition}_heatmap.npz', data=plotData)
    
plot_heatmap(3.1)
