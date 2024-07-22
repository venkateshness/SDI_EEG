#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from nilearn import datasets, maskers

HOMEDIR = "/users/local/Venkatesh/structure-function-eeg/"    
########################################################
#####movide/ freq bands stats#############################################
########################################################

video_watching_SDI_theta=np.log2(np.load(f"{HOMEDIR}/Generated_data/video1/Graph_SDI_related/empirical_SDI.npz")['theta'])
video_watching_SDI_alpha=np.log2(np.load(f"{HOMEDIR}/Generated_data/video1/Graph_SDI_related/empirical_SDI.npz")['alpha'])


atlas_yeo_2011 = datasets.fetch_atlas_yeo_2011()
yeo = atlas_yeo_2011.thick_7
glasser = f"{HOMEDIR}/src_data/Glasser_masker.nii.gz"

masker = maskers.NiftiMasker(standardize=False, detrend=False)
masker.fit(glasser)
glasser_vec = masker.transform(glasser)

yeo_vec = masker.transform(yeo)
yeo_vec = np.round(yeo_vec)

matches = []
match = []
best_overlap = []
for i, roi in enumerate(np.unique(glasser_vec)):
    overlap = []
    for roi2 in np.unique(yeo_vec):
        overlap.append(
            np.sum(yeo_vec[glasser_vec == roi] == roi2) / np.sum(glasser_vec == roi)
        )
    best_overlap.append(np.max(overlap))
    match.append(np.argmax(overlap))
    matches.append((i + 1, np.argmax(overlap)))


nw_theta=[]
nw_alpha=[]
nw_wideband=[]
nw_low_beta=[]
nw_high_beta=[]
nw_gamma=[]

video_watching_SDI_theta=np.log2(np.load(f"{HOMEDIR}/Generated_data/video1/Graph_SDI_related/empirical_SDI.npz")['theta'])
video_watching_SDI_alpha=np.log2(np.load(f"{HOMEDIR}/Generated_data/video1/Graph_SDI_related/empirical_SDI.npz")['alpha'])
video_watching_SDI_low_beta=np.log2(np.load(f"{HOMEDIR}/Generated_data/video1/Graph_SDI_related/empirical_SDI.npz")['low_beta'])
video_watching_SDI_high_beta=np.log2(np.load(f"{HOMEDIR}/Generated_data/video1/Graph_SDI_related/empirical_SDI.npz")['high_beta'])
video_watching_SDI_gamma=np.log2(np.load(f"{HOMEDIR}/Generated_data/video1/Graph_SDI_related/empirical_SDI.npz")['gamma'])
video_watching_SDI = np.log2(np.load(f"{HOMEDIR}/Generated_data/video1/Graph_SDI_related/empirical_SDI.npz")['wideband'])
for i in range(7):
    idx = np.array(match)==i
    nw_theta.append(np.mean(video_watching_SDI_theta[:, idx], axis=1))
    nw_alpha.append(np.mean(video_watching_SDI_alpha[:, idx], axis=1))
    nw_wideband.append(np.mean(video_watching_SDI[:, idx], axis=1))
    nw_low_beta.append(np.mean(video_watching_SDI_low_beta[:, idx], axis=1))
    nw_high_beta.append(np.mean(video_watching_SDI_high_beta[:, idx], axis=1))
    nw_gamma.append(np.mean(video_watching_SDI_gamma[:, idx], axis=1))



# Combine the lists into a single DataFrame
df = pd.DataFrame({
    'theta': nw_theta,
    'alpha': nw_alpha,
    'low_beta': nw_low_beta,
    'high_beta': nw_high_beta,
    'gamma': nw_gamma,
    'wideband': nw_wideband
})

# Get the list of network names
networks = df.columns.tolist()

# Initialize an empty DataFrame to store significance indicators
significance_df = pd.DataFrame(index=df.columns, columns=df.columns)

rho = [
]
p = []
for i in range(7):
    data_of_interest = df.iloc[i]
    for band1 in ['theta', 'alpha', 'low_beta', 'high_beta', 'gamma']:
        for band2 in ['theta', 'alpha', 'low_beta', 'high_beta', 'gamma']:
            if band1 != band2:
                
                stats, p_value = spearmanr(data_of_interest[band1], data_of_interest[band2])
                if p_value > 0.05:
                    significance_df.loc[band1, band2] = 0
                else:
                    significance_df.loc[band1, band2] = stats
                rho.append(stats)
                p.append(p_value)

        band_labels = [r'$\theta$', r'$\alpha$', r'low_$\beta$', r'high_$\beta$', r'$\gamma$']
        sns.heatmap(significance_df.astype(float), cmap='inferno', annot=True, fmt=".1f", xticklabels=band_labels, yticklabels=band_labels)
        plt.show()

