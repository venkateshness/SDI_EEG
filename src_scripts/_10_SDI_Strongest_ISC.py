#%%
import scipy.stats as stats
import numpy as np
import importlib
import os
import utility_functions
from nilearn.regions import signals_to_img_labels
from nilearn.datasets import fetch_icbm152_2009
from tqdm import tqdm

import _7_SDI_spatial_maps
import statsmodels.stats.multitest as multitest
import matplotlib.pyplot as plt

importlib.reload(utility_functions)


HOMEDIR = "/users/local/Venkatesh/structure-function-eeg/"
path_Glasser = f"{HOMEDIR}/src_data/Glasser_masker.nii.gz"
mnitemp = fetch_icbm152_2009()


graph  = np.load(f"{HOMEDIR}/src_data/individual_graphs.npz")

n_subjects = 43
n_roi = 360

def SDI_strongest_ISC(onset, band, n_surrogates=19):
    """SDI during strongest and weakest ISC periods
    Args:
        onset (list): list of the onset times for the strongest and weakest ISC
        band (str): the frequency band
        n_surrogates (int): number of surrogates
        
    Returns:
        data_for_weak_and_strong (dict): dictionary containing the SDI for the strongest and weakest ISC
        """
    data_for_weak_and_strong = dict()
    labels = ['strong', 'weak']
    for idx, weak_and_strong in enumerate(onset):
        if band=='theta':
            envelope_signal_bandpassed = np.load(f"{HOMEDIR}/Generated_data/video1/cortical_surface_related/{band}_bandpassed.npz")
        elif band=='widerband':
            envelope_signal_bandpassed = np.load(f"{HOMEDIR}/Generated_data/video1/cortical_surface_related/parcellated_widerband.npz")
        

        # surrogate SDI
        s_bundle = list()
        e_bundle = list()
        hf_bundle = list()
        lf_bundle = list()
        
        for subj, signal in envelope_signal_bandpassed.items():
            signal_sliced = signal[:, weak_and_strong:weak_and_strong+125]
            _, eigenvals, eigenvectors = utility_functions.eigmodes(graph[subj])
            _, lf_comp, hf_comp = utility_functions.fullpipeline(signal_sliced, eigenvectors, eigenvals)
            empirical_SDI = hf_comp/lf_comp
            e_bundle.append(empirical_SDI)
            hf_bundle.append(hf_comp)
            lf_bundle.append(lf_comp)
            surr_signal = utility_functions.surrogate_eigenmodes(eigenvectors, signal_sliced, n_surrogates)

            surr_SDI = list()
            for surr in tqdm(range(np.shape(surr_signal)[0])):
                graph_signal = np.squeeze(surr_signal[surr])

                lf, hf = utility_functions.fullpipeline(graph_signal, eigenvectors, eigenvals, is_surrogate=True)
                SDI = hf/lf
                surr_SDI.append(SDI)

            surr_SDI = np.squeeze(surr_SDI)
            s_bundle.append(surr_SDI)
            
        s_bundle_reshaped = np.swapaxes(s_bundle, 0, 1)

        empirical_SDI = np.array(hf_bundle)/np.array(lf_bundle)

        max_sdi_surr = np.max( s_bundle_reshaped, axis=0)
        min_sdi_surr = np.min( s_bundle_reshaped, axis=0)
        idx_max = empirical_SDI > max_sdi_surr
        idx_min = empirical_SDI < min_sdi_surr

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

        empi_SDI_avg = np.mean(empirical_SDI, axis = 0)
        empi_sig = empi_SDI_avg*bin_mask_one
        
        empi_sig[np.where(empi_sig==0)] = 1
        SDI_final = np.log2(empi_sig)
        # SDI_final[SDI_final<-1] = -1
        # SDI_final[SDI_final>1] = 1
        data_for_weak_and_strong[f'{labels[idx]}'] = SDI_final
        
        mnitemp = fetch_icbm152_2009()
        
        nifti = signals_to_img_labels(SDI_final, path_Glasser, mnitemp["mask"])
        _7_SDI_spatial_maps.customized_plotting_img_on_surf(stat_map=nifti, threshold=1e-20, cmap='cold_hot', views=["lateral", "medial"], hemispheres=["left", "right"], colorbar=False)
        plt.show()

    return data_for_weak_and_strong

corrca_ts_band_strong =np.load(f"{HOMEDIR}/Generated_data/video1/cortical_surface_related/ISC_bundle.npz")['theta']
data_for_theta = SDI_strongest_ISC([np.argmin(corrca_ts_band_strong[0, 5:165]), np.argmax(corrca_ts_band_strong[0, 5:165])] , 'theta')

corrca_ts_band_weak =np.load(f"{HOMEDIR}/Generated_data/video1/cortical_surface_related/ISC_bundle.npz")['widerband']
data_for_wideband = SDI_strongest_ISC([np.argmin(corrca_ts_band_weak[0, 5:165]), np.argmax(corrca_ts_band_weak[0, 5:165])] , 'widerband')

np.savez_compressed(f'{HOMEDIR}/Generated_data/Data_for_plots/SDI_strong_weak_ISC_theta.npz', **data_for_theta)
np.savez_compressed(f'{HOMEDIR}/Generated_data/Data_for_plots/SDI_strong_weak_ISC_widerband.npz', **data_for_wideband)


# %%

########################################################################
##################Strong ISC vs Weak ISC SDI comparison##################
########################################################################
def SDI_in_seconds(band):
    if band=='theta':
        envelope_bandpassed = np.load(f"{HOMEDIR}/Generated_data/video1/cortical_surface_related/{band}_bandpassed.npz")
    elif band=='widerband':
        envelope_bandpassed = np.load(f"{HOMEDIR}/Generated_data/video1/cortical_surface_related/parcellated_widerband.npz")
    
    lf_bundle = list()
    hf_bundle = list()

    #empirical SDI
    for sub, signal in envelope_bandpassed.items():
        connectome = graph[sub]
        _, eigenvals, eigenvectors = utility_functions.eigmodes(connectome)
        lf, hf = utility_functions.fullpipeline(signal, eigenvectors, eigenvals, in_seconds=True, is_surrogate=False)

        lf_bundle.append(lf)
        hf_bundle.append(hf)
        
    return lf_bundle, hf_bundle


for band in ['theta', 'widerband']:
    corrca_ts = np.load(f"{HOMEDIR}/Generated_data/video1/cortical_surface_related/ISC_bundle.npz")[f'{band}']

    lf_b, hf_b = SDI_in_seconds(f'{band}')

    lf_b_reshaped = np.reshape(lf_b, (43, 360, 170, 125))
    hf_b_reshaped = np.reshape(hf_b, (43, 360, 170, 125))

    lf_b_normed = np.linalg.norm(lf_b_reshaped, axis=3)
    hf_b_normed = np.linalg.norm(hf_b_reshaped, axis=3)

    SDI_seconds = np.log2(hf_b_normed/lf_b_normed)

    strong_time = np.argmax(corrca_ts_band_strong[0, 5:165]) #np.random.randint(5, 165)
    weak_time = np.argmin(corrca_ts_band_strong[0, 5:165]) #np.random.randint(5, 165)

    strong_ISC = SDI_seconds[:, :, strong_time]
    weak_ISC = SDI_seconds[:, :, weak_time]

    plt.show()
    
    obs_ttest_rel = stats.ttest_rel(strong_ISC, weak_ISC, axis=0)
    fdr = multitest.fdrcorrection(obs_ttest_rel.pvalue, alpha=0.05 )

    signal_to_plot = fdr[0]*obs_ttest_rel.statistic
    nifti = signals_to_img_labels(signal_to_plot, path_Glasser, mnitemp["mask"])
    _7_SDI_spatial_maps.customized_plotting_img_on_surf(stat_map=nifti, threshold=1e-20, cmap='cold_hot', views=["lateral", "medial"], hemispheres=["left", "right"], colorbar=False)
    plt.show()


