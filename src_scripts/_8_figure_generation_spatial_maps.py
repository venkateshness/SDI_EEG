
## Dedicated script to generate the spatial maps for the figures in the manuscript
# For more details about where the corresponding data are coming from, please refer to _8_figure_generation.py

from enigmatoolbox.plotting import plot_cortical
from nilearn import plotting
from nilearn.regions import signals_to_img_labels
from nilearn.datasets import fetch_icbm152_2009
import numpy as np
from nilearn import plotting
from nilearn.regions import signals_to_img_labels
from nilearn.datasets import fetch_icbm152_2009

HOMEDIR = "/users/local/Venkatesh/structure-function-eeg/"

labels_R =np.load(f'{HOMEDIR}/src_data/sourcespace_to_glasser_labels.npz')['labels_R']
labels_L =np.load(f'{HOMEDIR}/src_data/sourcespace_to_glasser_labels.npz')['labels_L']
labels_combined = np.concatenate((labels_L,labels_R))

path_Glasser = f"{HOMEDIR}/src_data/Glasser_masker.nii.gz"
mnitemp = fetch_icbm152_2009()



def expand(data):
    """Repopulate ROI-level data to vertex-level data for visualization"""
    data_fsa =np.zeros(20484)

    for i in range(360):
        data_fsa[np.where(labels_combined==i)] = data[i]
        data_fsa[np.where(labels_combined==i)] = data[i]
        
    return data_fsa


#### Figure 1 #####
## Change the value range for other Eigenvectors
eigenvectors = np.load(f'{HOMEDIR}/Data_for_plots/eigvecs.npz')['eigvecs']
for i in range(200, 201):
    eigenvectors_fs = expand(eigenvectors[:,i])
    plot_cortical( surface_name="fsa5",array_name=eigenvectors_fs, color_bar=False,  cmap='seismic', color_range=(np.min(eigenvectors_fs), -np.min(eigenvectors_fs)), filename=f'eigenvector{i+1}.png', screenshot=True, size=(3840, 2160))

#### Figure 2 #####
# Panel a)
activity_movie = np.load(f'{HOMEDIR}/Data_for_plots/activity_avged_video.npz')['activity_avged_video']
activity_movie_zscore = (activity_movie - np.mean(activity_movie, axis=0))/np.std(activity_movie, axis=0)

activity_movie_zscore[np.where(activity_movie_zscore>2.33)] = 2.33
activity_movie_zscore[np.where(activity_movie_zscore<-2.33)] = -2.33
activity_movie_zscore_fs = expand(activity_movie_zscore)

plot_cortical( surface_name="fsa5",array_name=activity_movie_zscore_fs, color_bar=False,  cmap='PiYG_r', screenshot=False, size=(3840, 2160), color_range=(-2.52, 2.52), interactive=False, scale=(1,1))

#### Figure 2 #####
# Panel a)
activity_rest = np.load(f'{HOMEDIR}/Data_for_plots/activity_avged_rest.npz')['activity_avged_rest']
activity_rest_zscore = (activity_rest - np.mean(activity_rest, axis=0))/np.std(activity_rest, axis=0)

activity_rest_zscore[np.where(activity_rest_zscore>2.33)] = 2.33
activity_rest_zscore[np.where(activity_rest_zscore<-2.33)] = -2.33
activity_rest_zscore_fs = expand(activity_rest_zscore)
plot_cortical( surface_name="fsa5",array_name=activity_rest_zscore_fs, color_bar=False,  cmap='PiYG_r', screenshot=False, size=(3840, 2160), color_range=(-2.52, 2.52))


#####Figure 2#####
# Panel c)
empi_SDI_movie = np.load(f'{HOMEDIR}/Data_for_plots/grouplevel_SDI_video1.npz')['widerband']
empi_SDI_movie[np.where(empi_SDI_movie<-1)] = -1
empi_SDI_movie[np.where(empi_SDI_movie>1)] = 1
empi_SDI_movie_fs = expand(empi_SDI_movie)
plot_cortical( surface_name="fsa5",array_name=empi_SDI_movie_fs, color_bar=False,  cmap='seismic', color_range=(-1,1), screenshot=False, size=(3840, 2160), scale=(1,1))



#### Figure 2 #####
# Panel d)
empi_SDI_rest=np.load(f'{HOMEDIR}/Data_for_plots/grouplevel_SDI_rest.npz')['widerband']
empi_SDI_rest[np.where(empi_SDI_rest<-1)] = -1
empi_SDI_rest[np.where(empi_SDI_rest>1)] = 1
empi_SDI_rest_fs = expand(empi_SDI_rest)
plot_cortical( surface_name="fsa5",array_name=empi_SDI_rest_fs, color_bar=False,  cmap='seismic', color_range=(-1,1),  screenshot=True, size=(3840, 2160), scale=(1,1))


####Figure 2 #####
# Panel e)
movie_vs_rest = np.load(f'{HOMEDIR}/Data_for_plots/SDI_corrected_movie_vs_rest_widerband.npz')['widerband']

movie_vs_rest[np.where(movie_vs_rest<-5)] = -5
movie_vs_rest[np.where(movie_vs_rest>5)] = 5
movie_vs_rest_fs = expand(movie_vs_rest)
plot_cortical( surface_name="fsa5",array_name=movie_vs_rest_fs, color_bar=False,  cmap='seismic', color_range=(-5,5), screenshot=False, size=(3840, 2160), scale=(1,1))

#### Figure 3 #####
# First 4 columns

bands = ['theta','alpha', 'low_beta', 'high_beta', 'gamma']

for band in bands:
    empi_SDI_movie = np.load(f'{HOMEDIR}/Data_for_plots/grouplevel_SDI_video1.npz')[f'{band}']
    empi_SDI_movie[np.where(empi_SDI_movie<-1)] = -1
    empi_SDI_movie[np.where(empi_SDI_movie>1)] = 1
    empi_SDI_movie_fs = expand(empi_SDI_movie)
    plot_cortical( surface_name="fsa5",array_name=empi_SDI_movie_fs, color_bar=False,  cmap='seismic', color_range=(-1,1), screenshot=False, size=(3840, 2160), scale=(1,1))

# Last column
path_Glasser = f"{HOMEDIR}/structure-function-eeg/Glasser_masker.nii.gz"
mnitemp = fetch_icbm152_2009()

bands = ['theta','alpha', 'low_beta', 'high_beta', 'gamma']

for band in bands:
    SDI_movie_vs_rest = np.load(f"{HOMEDIR}/Data_for_plots/SDI_corrected_movie_vs_rest_wideband.npz")[f'{band}']
    SDI_movie_vs_rest[np.where(SDI_movie_vs_rest<-5)] = -5
    SDI_movie_vs_rest[np.where(SDI_movie_vs_rest>5)] = 5

    nifti= signals_to_img_labels(SDI_movie_vs_rest, path_Glasser, mnitemp["mask"])
    plotting.plot_glass_brain(nifti, cmap='coolwarm', display_mode='r', symmetric_cbar=True,vmin=-5, vmax=5, colorbar=False, plot_abs=False)
    

##### Figure 4 #####
# Panel a)
empirical_SDI_anvideo = np.load(f"{HOMEDIR}/Data_for_plots/grouplevel_SDI_video2.npz")['widerband']

empirical_SDI_anvideo[np.where(empirical_SDI_anvideo<-1)] = -1
empirical_SDI_anvideo[np.where(empirical_SDI_anvideo>1)] = 1
empirical_SDI_anvideo_fs = expand(empirical_SDI_anvideo)
nifti= signals_to_img_labels(empirical_SDI_anvideo_fs, path_Glasser, mnitemp["mask"])    
plot_cortical( surface_name="fsa5",array_name=empirical_SDI_anvideo_fs, color_bar=False,  cmap='seismic', screenshot=True, size=(3840, 2160), color_range=(-1,1))

# Panel b)
empirical_SDI_anvideo_vs_rest = np.load(f"{HOMEDIR}Data_for_plots/SDI_corrected_anvideo_vs_rest_widerband.npz")['widerband']
empirical_SDI_anvideo_vs_rest[np.where(empirical_SDI_anvideo_vs_rest<-5.49)] = -5.49
empirical_SDI_anvideo_vs_rest[np.where(empirical_SDI_anvideo_vs_rest>5.49)] = 5.49
empirical_SDI_anvideo_vs_rest_fs = expand(empirical_SDI_anvideo_vs_rest)
plot_cortical( surface_name="fsa5",array_name=empirical_SDI_anvideo_vs_rest_fs, color_bar=False,  cmap='seismic', screenshot=True, size=(3840, 2160), color_range=(-5.49,5.49))

# Supplementary Figure 1
# RIght side of the panels - Spatial maps

def strong_and_weak(band, condition):
    ISC = np.load(f"{HOMEDIR}/Data_for_plots/SDI_strong_weak_ISC_{band}.npz", allow_pickle=True)[f'{condition}']
    ISC[np.where(ISC<-1)] = -1
    ISC[np.where(ISC>1)] = 1
    ISC_fs = expand(ISC)
    plot_cortical( surface_name="fsa5",array_name=ISC_fs, color_bar=False,  cmap='seismic', screenshot=True, size=(3840, 2160), color_range=(-1,1))


strong_and_weak('theta', 'strong')
strong_and_weak('theta', 'weak')
strong_and_weak('widerband', 'strong')
strong_and_weak('widerband', 'weak')
