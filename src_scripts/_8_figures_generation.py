#%%
import numpy as np
import utility_functions
import importlib
import matplotlib.pyplot as plt
from nilearn.datasets import fetch_icbm152_2009
from nilearn import image
from nilearn.regions import signals_to_img_labels
import _7_SDI_spatial_maps
import _6_SDI_statistics
import scipy.stats as stats
import statsmodels.stats.multitest as smt
import os

importlib.reload(_6_SDI_statistics)
importlib.reload(_7_SDI_spatial_maps)
importlib.reload(utility_functions)

mnitemp = fetch_icbm152_2009()
HOMEDIR = "/users/local/Venkatesh/structure-function-eeg/"
graph  = np.load(f"{HOMEDIR}/src_data/individual_graphs.npz")
path_Glasser = f"{HOMEDIR}/src_data/Glasser_masker.nii.gz"

plt.style.use('seaborn-whitegrid')

# Customize font settings
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'font.family': 'serif',  # Choose a serif font for better readability
    'font.serif': ['Times New Roman'],
})



#%%
# Connectome
################################################
################Figure 1######################
################################################

## Connectome matrix/ Panel a)
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

adj_mat =list()
for i in range(43):
    adj_mat.append(graph[graph.files[i]])
avg_connectome = np.mean(adj_mat, axis=0)



# cbar.ax.tick_params(labelsize=20)  # Adjust the label size


laplacian, eigvals, eigvecs = utility_functions.eigmodes(avg_connectome)
plt.grid(False, linestyle='--', alpha=0.7, linewidth=0.5)  # Adjust the linewidth here
ax.imshow(np.log1p(avg_connectome), cmap='jet')
fig.savefig(f'{HOMEDIR}/Results/Figure_1/avg_connectome.svg', dpi=300)
plt.figure(figsize=(6, 6), dpi=300)

plt.plot(eigvals, color='blue', linewidth=2)

plt.grid(False, linestyle='--', alpha=0.7, linewidth=0.5)  # Adjust the linewidth here

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# plt.savefig(f'{HOMEDIR}/Results/Figure_1/eigenvalues_plot.svg', bbox_inches='tight')

# Display the plot
plt.show()
# np.savez_compressed(f'{HOMEDIR}/Generated_data/Data_for_plots/eigvecs.npz', eigvecs=eigvecs)

#%%
#Fig1 / Panel c

fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
plt.style.use('seaborn-whitegrid')

# Customize font settings
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'font.family': 'serif',  # Choose a serif font for better readability
    'font.serif': ['Times New Roman'],
})


envelope_signal_bandpassed = np.load(f"{HOMEDIR}/Generated_data/video1/cortical_surface_related/wideband_bandpassed.npz")
gpsd_bundle = list()

for sub, data in envelope_signal_bandpassed.items():
    gpsd_avg,_ = utility_functions.compute_gpsd(data, eigvecs)
    gpsd_bundle.append(gpsd_avg)

fig, ax = plt.subplots(figsize=(10,10), dpi = 300)
plt.semilogy(eigvals, np.mean(gpsd_bundle,axis=0), color='blue', linewidth=2)
plt.axvspan(0, 0.1379, color='teal', alpha=0.35)
plt.axvspan(0.1379, np.max(eigvals), color='magenta', alpha=0.35)
fig.savefig(f'{HOMEDIR}/Generated_data/Results/Figure_1/PSD_plot.svg', bbox_inches='tight')

# %%
################################################################    
################Figure 2################################
################################################################
        

###################################################################
#######Averaged cortical activity / Panel a)#######################
###################################################################
video_activity = list(np.load(f"{HOMEDIR}/Generated_data/video1/cortical_surface_related/parcellated_widerband.npz").values())
resting_state = list(np.load(f"{HOMEDIR}/Generated_data/rest/cortical_surface_related/parcellated_widerband.npz").values())

def plot_activity(activity, video=True):
    activity_avged = np.mean(activity, axis=(0, 2))
    nifti= signals_to_img_labels(activity_avged, path_Glasser, mnitemp["mask"])

    if video:
        np.savez_compressed(f"{HOMEDIR}/Generated_data/Data_for_plots/activity_avged_video.npz", activity_avged_video=activity_avged)
    else:
        np.savez_compressed(f"{HOMEDIR}/Generated_data/Data_for_plots/activity_avged_rest.npz", activity_avged_rest=activity_avged)
    # _7_SDI_spatial_maps.customized_plotting_img_on_surf(stat_map=nifti, vmin=np.min(activity_avged), vmax=np.max(activity_avged), views=["lateral"], hemispheres=["left", "right"], symmetric_cbar=False, colorbar=False, cmap="inferno", output_file=f'{HOMEDIR}/Generated_data/Results/Figure_2/activity_video.svg')

plot_activity(video_activity, video=True)
plot_activity(resting_state, video=False)

#%%
#################################################################
#######Averaged Spectrum / Panel b)##############################
#################################################################

def gpsd_avg_bundle(activity):
    gpsd_avg_bundle = list()
    adj_mat = list()

    for i in range(43):
        adj_mat.append(graph[graph.files[i]])
    avg_connectome = np.mean(adj_mat, axis=0)
    
    for i in range(43):
        _, eigvals, eigvecs = utility_functions.eigmodes(avg_connectome)
        gpsd_avg,_ = utility_functions.compute_gpsd(activity[i], eigvecs)
        gpsd_avg_bundle.append(gpsd_avg)

    critical_freq = utility_functions.split_gpsd(np.mean(gpsd_avg_bundle, axis=0), eigvals)
    return critical_freq, gpsd_avg_bundle

envelope_signal_bandpassed = list(np.load(f"{HOMEDIR}/Generated_data/rest/cortical_surface_related/parcellated_widerband.npz").values())
rest_normed = (envelope_signal_bandpassed - np.mean(envelope_signal_bandpassed, axis=2)[:,:, np.newaxis])/np.std(envelope_signal_bandpassed, axis=2)[:,:, np.newaxis]

envelope_signal_bandpassed = list(np.load(f"{HOMEDIR}/Generated_data/video1/cortical_surface_related/parcellated_widerband.npz").values())
video_normed = (envelope_signal_bandpassed - np.mean(envelope_signal_bandpassed, axis=2)[:,:, np.newaxis])/np.std(envelope_signal_bandpassed, axis=2)[:,:, np.newaxis]

_, gpsd_avg_bundle_movie = gpsd_avg_bundle(video_normed)
_, gpsd_avg_bundle_rest = gpsd_avg_bundle(rest_normed)
#%%

fig, ax = plt.subplots(figsize=(25,10))
plt.semilogy(eigvals, np.mean(gpsd_avg_bundle_movie, axis=0), label = 'Video', alpha=0.9, c='magenta')
plt.semilogy(eigvals, np.array(gpsd_avg_bundle_movie).T, alpha=0.1, c='magenta')

plt.semilogy(eigvals, np.array(gpsd_avg_bundle_rest).T, alpha=0.1, c='olive')
plt.semilogy(eigvals, np.mean(gpsd_avg_bundle_rest, axis=0), label = 'Rest', alpha=0.9, c='olive')
plt.legend()
plt.grid(False, which='major', linestyle='--', linewidth=0.5)

plt.tight_layout()
fig.savefig(f'{HOMEDIR}/Results/Figure_2/PSD_plot.svg', bbox_inches='tight')
#%%


###################
#####Panel C)######
######################

########################
#######Video and Rest###
########################


# grouplevel_SDI_video1=_6_SDI_statistics.stats_full_test(bands=['theta', 'alpha', 'low_beta', 'high_beta', 'gamma',  'widerband'], condition='video1')
grouplevel_SDI_rest=_6_SDI_statistics.stats_full_test(bands=['theta', 'alpha', 'low_beta', 'high_beta', 'gamma',  'widerband'], condition='rest')
# grouplevel_SDI_video2=_6_SDI_statistics.stats_full_test(bands=['theta', 'alpha', 'low_beta', 'high_beta', 'gamma', 'widerband'], condition='video2')


# np.savez_compressed(f"{HOMEDIR}/Generated_data/Data_for_plots/grouplevel_SDI_video1.npz", **grouplevel_SDI_video1)
# np.savez_compressed(f"{HOMEDIR}/Generated_data/Data_for_plots/grouplevel_SDI_rest.npz", **grouplevel_SDI_rest)
# np.savez_compressed(f"{HOMEDIR}/Generated_data/Data_for_plots/grouplevel_SDI_video2.npz", **grouplevel_SDI_video2)

#%%


########################
#######Video vs Rest####
########################
SDI_corrected_movie_vs_rest_all_band = dict()

for band in [ 'theta', 'alpha', 'low_beta', 'high_beta', 'gamma', 'widerband']:#'theta', 'alpha', 'low_beta', 'high_beta', 'gamma', 'wideband',
    video_watching_SDI=np.log2(np.load(f"{HOMEDIR}/revision/Generated_data_revision/video1/Graph_SDI_related_no_envelope_signal/empirical_SDI.npz")[f'{band}'])
    rs_SDI = np.log2(np.load(f"{HOMEDIR}/revision/Generated_data_revision/rest/Graph_SDI_related_no_envelope_signal/empirical_SDI.npz")[f'{band}'])

    pvals = []
    tvals = []
    for i in range(360):
        ttest = stats.ttest_rel(video_watching_SDI[:,i], rs_SDI[:,i])
        pvals.append(ttest.pvalue)
        tvals.append(ttest.statistic)
    
    signal_for_uncorrected = (np.array(pvals)<0.05)*tvals
    #Corrected
    pvals_corrected = np.array(smt.fdrcorrection(pvals, alpha=0.05))
    SDI_corrected_movie_vs_rest = pvals_corrected[0]*tvals
    nifti= signals_to_img_labels(SDI_corrected_movie_vs_rest, path_Glasser, mnitemp["mask"])
    # plotting.plot_glass_brain(nifti, cmap='seismic', symmetric_cbar=True, vmin=-5, vmax=5, colorbar=True)
    _7_SDI_spatial_maps.customized_plotting_img_on_surf(stat_map=nifti, threshold=1e-20, vmin = np.min(SDI_corrected_movie_vs_rest), vmax = -np.min(SDI_corrected_movie_vs_rest), cmap='cold_hot', views=["lateral", "medial"], hemispheres=["left", "right"], colorbar=False)
    
    plt.show()
    SDI_corrected_movie_vs_rest_all_band[f'{band}'] = SDI_corrected_movie_vs_rest
# np.savez_compressed(f"{HOMEDIR}/Generated_data/Data_for_plots/SDI_corrected_movie_vs_rest_{band}.npz", **SDI_corrected_movie_vs_rest_all_band)

# %%
########################
#######Video vs Rest Consensus#### 
########################

# grouplevel_SDI_video1_consensus=_6_SDI_statistics.stats_full_test(bands=['theta', 'alpha', 'low_beta', 'high_beta', 'gamma', 'wideband'], condition='movie_consensus')
# grouplevel_SDI_rest_consensus=_6_SDI_statistics.stats_full_test(bands=['theta', 'alpha', 'low_beta', 'high_beta', 'gamma', 'wideband'], condition='rest_consensus')

# np.savez_compressed(f"{HOMEDIR}/Generated_data/Data_for_plots/grouplevel_SDI_video1_consensus.npz", **grouplevel_SDI_video1_consensus)
# np.savez_compressed(f"{HOMEDIR}/Generated_data/Data_for_plots/grouplevel_SDI_rest_consensus.npz", **grouplevel_SDI_rest_consensus)

#%%

####################
##### Fig 4##########
####################

################################################
#######Video vs Rest//Retest reliability####
################################################


for band in ['widerband']:
    video_watching_SDI=np.log2(np.load(f"{HOMEDIR}/Generated_data/video2/Graph_SDI_related/empirical_SDI.npz")[f'{band}'])
    rs_SDI = np.log2(np.load(f"{HOMEDIR}/Generated_data/rest/Graph_SDI_related/empirical_SDI.npz")[f'{band}'])


    pvals = []
    tvals = []
    for i in range(360):
        ttest = stats.ttest_rel(video_watching_SDI[:,i], rs_SDI[:,i])
        pvals.append(ttest.pvalue)
        tvals.append(ttest.statistic)

    
    pvals_corrected = np.array(smt.fdrcorrection(pvals, alpha=0.05))

    SDI_corrected_anvideo_vs_rest = pvals_corrected[0]*tvals
    nifti= signals_to_img_labels(SDI_corrected_anvideo_vs_rest, path_Glasser, mnitemp["mask"])
    # nifti.to_filename(f"{HOMEDIR}/Generated_data/movie/Graph_SDI_related/empirical_SDI_{band}.nii.gz")
    _7_SDI_spatial_maps.customized_plotting_img_on_surf(stat_map=nifti, threshold=1e-20, vmin = np.min(SDI_corrected_anvideo_vs_rest), vmax = -np.min(SDI_corrected_anvideo_vs_rest), cmap='cold_hot', views=["lateral", "medial"], hemispheres=["left", "right"], colorbar=False)
    # plt.show()
    np.savez_compressed(f"{HOMEDIR}/Generated_data/Data_for_plots/SDI_corrected_anvideo_vs_rest_{band}.npz", widerband=SDI_corrected_anvideo_vs_rest)


#%%

###########SDI during Strong ISC and Weak ISC################

# Customize font settings
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'font.family': 'serif',  # Choose a serif font for better readability
    'font.serif': ['Times New Roman'],
})


# Loop through bands
for band in ['theta', 'widerband']:
    # Load data (replace with your actual data path)
    corrca_ts = np.load(f"{HOMEDIR}/Generated_data/video1/cortical_surface_related/ISC_bundle.npz")[band]

    # Create a new figure
    plt.figure(figsize=(8, 6))

    # Plot the time series
    plt.plot(corrca_ts[0], linewidth=2.5, color='blue')  # Replace 'Your Label' with an appropriate label

    # Add vertical lines
    max_index = np.argmax(corrca_ts[0, 1:165]) + 1
    min_index = np.argmin(corrca_ts[0, 1:165]) + 1

    plt.axvline(max_index, color='black', linestyle='--', label=f'Strong ISC', linewidth=2)
    plt.axvline(min_index, color='teal', linestyle='--', label=f'Weak ISC', linewidth=2)

    # Customize legend
    plt.legend(fontsize=12)

    # Set grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Save or show the plot
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.savefig(f"{HOMEDIR}/Results/Figure_7/SDI_strongest_ISC_{band}.svg", dpi=300)  
    plt.show()  # Show the figure
#%%

import pandas as pd
df = pd.read_excel('/users/local/Venkatesh/structure-function-eeg/src_data/Glasser_2016_Table.xlsx', header =1)
area_description = df.columns[2]
area_description_data = df[area_description]
area_description_data_concat = np.concatenate([area_description_data.values] * 2, axis=0)

#%%
condition="video1"
empi_SDI = np.log2(np.squeeze(np.load(HOMEDIR + f"Generated_data/{condition}/Graph_SDI_related/empirical_SDI.npz")[f'widerband']))
signal = np.mean(empi_SDI, axis=0)


def sort_index(lst, rev=True):
    index = range(len(lst))
    s = sorted(index, reverse=rev, key=lambda i: lst[i])


    return s

returned_indices = sort_index(signal)

#%%

sorted(zip(signal, area_description_data_concat), reverse=True)[:3]

# %%
