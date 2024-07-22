
import numpy as np
import utility_functions
import importlib
import matplotlib.pyplot as plt
from nilearn.datasets import fetch_icbm152_2009
from nilearn import image
from nilearn.regions import signals_to_img_labels
import _6_SDI_statistics
import scipy.stats as stats
import statsmodels.stats.multitest as smt
import seaborn as sns
import pandas as pd

importlib.reload(_6_SDI_statistics)
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



# Connectome
################################################
################Figure 1######################
################################################

######################### Connectome matrix/ Panel a)#######################
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
fig.savefig(f'{HOMEDIR}/Results/Figure_1/avg_connectome.svg', dpi=300)


######################### Graph Spectrum (Eigenvalue distribution)/ Panel b)#######################
adj_mat =list()
for i in range(43):
    adj_mat.append(graph[graph.files[i]])
avg_connectome = np.mean(adj_mat, axis=0)


laplacian, eigvals, eigvecs = utility_functions.eigmodes(avg_connectome)
plt.grid(False, linestyle='--', alpha=0.7, linewidth=0.5) 
ax.imshow(np.log1p(avg_connectome), cmap='jet')

plt.figure(figsize=(6, 6), dpi=300)
plt.plot(eigvals, color='blue', linewidth=2)
plt.grid(False, linestyle='--', alpha=0.7, linewidth=0.5)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.savefig(f'{HOMEDIR}/Results/Figure_1/eigenvalues_plot.svg', bbox_inches='tight')
plt.show()
# np.savez_compressed(f'{HOMEDIR}/Generated_data/Data_for_plots/eigvecs.npz', eigvecs=eigvecs)


#####################Graph Power Spectrum, Panel c)#######################
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
# fig.savefig(f'{HOMEDIR}/Generated_data/Results/Figure_1/PSD_plot.svg', bbox_inches='tight')

###################note: Fig 1; Panel d, generated using _8.figures_generation_spatial_map.py script ############

################################################################    
################Figure 2################################
################################################################
        

#######Averaged cortical activity / Panel a)#######################

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
###################note: Fig 2; Panel a, figures generated using _8.figures_generation_spatial_map.py script ############




##############################Fig 2 Panel b)##############################

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


fig, ax = plt.subplots(figsize=(25,10))
plt.semilogy(eigvals, np.mean(gpsd_avg_bundle_movie, axis=0), label = 'Video', alpha=0.9, c='magenta')
plt.semilogy(eigvals, np.array(gpsd_avg_bundle_movie).T, alpha=0.1, c='magenta')

plt.semilogy(eigvals, np.array(gpsd_avg_bundle_rest).T, alpha=0.1, c='olive')
plt.semilogy(eigvals, np.mean(gpsd_avg_bundle_rest, axis=0), label = 'Rest', alpha=0.9, c='olive')
plt.legend()
plt.grid(False, which='major', linestyle='--', linewidth=0.5)

plt.tight_layout()
# fig.savefig(f'{HOMEDIR}/Results/Figure_2/PSD_plot.svg', bbox_inches='tight')


############################Figure 2 Panel c, d & Figure 3 all panel, first 4 columns #############################
#Note: The code below exports data for 
#the statistically thresholded spatial maps for Wideband (figure 2 panel c & d)
#as well as the spatial maps for the first 4 columns of figure 3
#exported data are used by the  script to generate the figures

grouplevel_SDI_video1=_6_SDI_statistics.stats_full_test(bands=['theta', 'alpha', 'low_beta', 'high_beta', 'gamma', 'wideband', 'widerband'], condition='video1')
grouplevel_SDI_rest=_6_SDI_statistics.stats_full_test(bands=['theta', 'alpha', 'low_beta', 'high_beta', 'gamma', 'wideband', 'widerband'], condition='rest')


np.savez_compressed(f"{HOMEDIR}/Generated_data/Data_for_plots/grouplevel_SDI_video1.npz", **grouplevel_SDI_video1)
np.savez_compressed(f"{HOMEDIR}/Generated_data/Data_for_plots/grouplevel_SDI_rest.npz", **grouplevel_SDI_rest)
###################note: Fig 2; Panel c, d generated using _8.figures_generation_spatial_map.py script ############


############################Figure 2 Panel e) #############################

SDI_corrected_movie_vs_rest_all_band = dict()

for band in [ 'widerband']:#'theta', 'alpha', 'low_beta', 'high_beta', 'gamma', 'wideband',
    video_watching_SDI=np.log2(np.load(f"{HOMEDIR}/Generated_data/video1/Graph_SDI_related/empirical_SDI.npz")[f'{band}'])
    rs_SDI = np.log2(np.load(f"{HOMEDIR}/Generated_data/rest/Graph_SDI_related/empirical_SDI.npz")[f'{band}'])

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
    _7_SDI_spatial_maps.customized_plotting_img_on_surf(stat_map=nifti, threshold=1e-20, vmin = np.min(SDI_corrected_movie_vs_rest), vmax = -np.min(SDI_corrected_movie_vs_rest), cmap='cold_hot', views=["lateral", "medial"], hemispheres=["left", "right"], colorbar=False)
    
    plt.show()
    SDI_corrected_movie_vs_rest_all_band[f'{band}'] = SDI_corrected_movie_vs_rest
# np.savez_compressed(f"{HOMEDIR}/Generated_data/Data_for_plots/SDI_corrected_movie_vs_rest_{band}.npz", **SDI_corrected_movie_vs_rest_all_band)
###################note: Fig 2; Panel e generated using _8.figures_generation_spatial_map.py script ############


##### Figure  4##########

#Panel a)
grouplevel_SDI_video2=_6_SDI_statistics.stats_full_test(bands=['widerband'], condition='video2') # ['theta', 'alpha', 'low_beta', 'high_beta', 'gamma', 'wideband', 'widerband']
np.savez_compressed(f"{HOMEDIR}/Generated_data/Data_for_plots/grouplevel_SDI_video2.npz", **grouplevel_SDI_video2)


#Panel b)
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
    # np.savez_compressed(f"{HOMEDIR}/Generated_data/Data_for_plots/SDI_corrected_anvideo_vs_rest_{band}.npz", widerband=SDI_corrected_anvideo_vs_rest)



#####Figure 5 Heatmaps##########
#####Spatial maps are generated using the  script (Fig 2, Panel c &d; Fig 4 panel a) 
plotData = np.load("/users/local/Venkatesh/structure-function-eeg/Data_for_plots/decoding_video1_heatmap.npz")['data']
df = pd.DataFrame(plotData)
plt.style.use('fivethirtyeight')

index = (['Visual_cortex_sensory', 'Motor_cortex_hand', 'Auditory_speech_temporal','Pain_somatosensory_stimulation', 'Motion_perception_visual', 
'Face_faces_facial', 'Memory_working_wm', 'Response_inhibition_control', 'Language_reading_word', 'Attention_attentional_target',
'Number_ips_numerical', 'Action_actions_observation', 'Reward_feedback_striatum', 'Control_conflict_task', 'Decision_making_risk', 
'Social_empathy_moral', 'Emotional_amygdala_negative', 'Imagery_mental_events', 'Memory_retrieval_encoding', 'Mpfc_social_medial']) # Manual ordering; rationale explained in the Section 2.4 (Decoding of the SDI maps)

df.index = index
columns = [str(i*10)+'-'+str((i*10)+10) for i in range(10)]
df.columns = columns

max_ = 12.45
min_ = 3.1
sns.set(context="paper", font= 'sans-serif', font_scale=5)
f, (ax1) = plt.subplots(nrows=1,ncols=1,figsize=(25, 25), sharey=True)
cax = sns.heatmap(df, linewidths=1, square=False, cmap='RdPu', robust=False, 
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


####Supplementary figure 1##########
# Corresponding Spatial maps generated using the _8_figures_generation_spatial_maps.py script

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

    plt.legend(fontsize=12)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    # plt.savefig(f"{HOMEDIR}/Results/Figure_7/SDI_strongest_ISC_{band}.svg", dpi=300)  
    plt.show()
