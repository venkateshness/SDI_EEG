#%%
import pingouin as pg
import numpy as np
import pandas as pd

HOMEDIR = "/users/local/Venkatesh/structure-function-eeg/"

def ICC_compute(condition1, condition2):
    group1 = condition1
    group2 = condition2

    data_for = {'1': group1, '2': group2}

    data = pd.DataFrame(data_for)

    melted_data = data.melt(value_vars=['1', '2'], var_name='Group', value_name='Score')
    melted_data['regions']= np.hstack([list(range(1, 361)), list(range(1, 361))])

    icc_result = pg.intraclass_corr(data=melted_data, targets='regions', raters='Group', ratings='Score')
    
    return icc_result

##############
##ICC between Video 1 and Video 2/ Fig2 panel c and Fig 4 panel a
##############
signal_for_corrected_movie = np.load(f"{HOMEDIR}/Generated_data/Data_for_plots/grouplevel_SDI_video2.npz")['widerband']
signal_for_corrected_anvideo = np.load(f"{HOMEDIR}/Generated_data/Data_for_plots/grouplevel_SDI_rest.npz")['widerband']
icc_video1_and_video2 = ICC_compute(signal_for_corrected_movie, signal_for_corrected_anvideo)

##############
##ICC between contrast maps. Fig2 panel e and Fig 4 panel b
##############


video1_vs_rest = np.load(f"{HOMEDIR}/Generated_data/Data_for_plots/SDI_corrected_movie_vs_rest_widerband.npz")['widerband']
video2_vs_rest = np.load(f"{HOMEDIR}/Generated_data/Data_for_plots/SDI_corrected_anvideo_vs_rest_widerband.npz")['widerband']
ICC_contrast = ICC_compute(video1_vs_rest, video2_vs_rest)

# %%
icc_video1_and_video2
# %%
ICC_contrast
# %%
