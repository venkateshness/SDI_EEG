#%%
import numpy as np
from nilearn.regions import signals_to_img_labels
from nilearn.datasets import fetch_icbm152_2009
import scipy.stats as stats
from nilearn import plotting
HOMEDIR = "/users/local/Venkatesh/structure-function-eeg/"
import matplotlib.pyplot as plt

n_subjects = 43
n_events = 85
n_roi = 360
n_surrogate = 19
SDI_movie = [] 
SDI_anvideo = []


def stats_full_test(bands, condition):
    SDI = dict()
    
    for band in bands:
        
        empi_SDI = np.squeeze(np.load(HOMEDIR + f"Generated_data_revision/{condition}/Graph_SDI_related/empirical_SDI.npz")[f'{band}'])
        surrogate_SDI =np.load(f'{HOMEDIR}/Generated_data_revision/{condition}/Graph_SDI_related/surrogate_SDI.npz')[f'{band}']
        
        max_sdi_surr = np.max( surrogate_SDI, axis=0)
        min_sdi_surr = np.min( surrogate_SDI, axis=0)
        idx_max = empi_SDI > max_sdi_surr
        idx_min = empi_SDI < min_sdi_surr
        detect_max = np.sum(idx_max, axis=0)
        detect_min = np.sum(idx_min, axis=0)

        x =range(101)
        y = stats.binom.sf(x, 100, p = 0.05)
        THRsubjs = np.min(np.where(y<0.05/360))
        THRsubjs = np.floor((n_subjects)/100*THRsubjs)+1
        SDI_high = np.where(detect_max>THRsubjs)
        SDI_low = np.where(detect_min>THRsubjs)
        regions_sig = np.unique(sorted(np.hstack([np.array(SDI_high), np.array(SDI_low)])))
        bin_mask_one = np.zeros((n_roi,))
        bin_mask_one[regions_sig] = 1

        empi_SDI_avg = np.mean(empi_SDI, axis = 0)
        empi_sig = empi_SDI_avg*bin_mask_one
        empi_sig[np.where(empi_sig==0)] = 1

        SDI_final = np.log2(empi_sig)
        # SDI_anvideo.append(SDI_final)

        SDI_final[SDI_final<-1] = -1
        SDI_final[SDI_final>1] = 1
        
        SDI[f"{band}"] = SDI_final
        path_Glasser = f"{HOMEDIR}/src_data/Glasser_masker.nii.gz"
        mnitemp = fetch_icbm152_2009()
        
        nifti = signals_to_img_labels(SDI_final, path_Glasser, mnitemp["mask"])
        if band == 'wideband':
            nifti.to_filename(f"{HOMEDIR}/Generated_data_revision/{condition}/Graph_SDI_related/SDI_{band}_{condition}.nii.gz")
        if band == 'widerband':
            nifti.to_filename(f"{HOMEDIR}/Generated_data_revision/{condition}/Graph_SDI_related/SDI_{band}_{condition}.nii.gz")
        
        # _7_SDI_spatial_maps.customized_plotting_img_on_surf(stat_map=nifti, threshold=1e-20, vmin=-1, vmax=1, cmap='cold_hot', views=["lateral", "medial"], hemispheres=["left", "right"], colorbar=False)
        plotting.plot_img_on_surf(stat_map=nifti, vmin=-1, vmax=1, threshold=0.0001, title=f'{condition}/ wideband no-envelope')
        
        
    return SDI
# %%

stats_full_test(["widerband"], 'video2')
# %%
