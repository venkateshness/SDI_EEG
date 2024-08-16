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

video_watching_SDI_theta=np.log2(np.load(f"{HOMEDIR}/revision/Generated_data_revision/video1/Graph_SDI_related/empirical_SDI_shuffled_.npz")['theta'])
video_watching_SDI_alpha=np.log2(np.load(f"{HOMEDIR}/revision/Generated_data_revision/video1/Graph_SDI_related/empirical_SDI.npz")['alpha'])
video_watching_SDI_low_beta=np.log2(np.load(f"{HOMEDIR}/revision/Generated_data_revision/video1/Graph_SDI_related/empirical_SDI.npz")['low_beta'])
video_watching_SDI_high_beta=np.log2(np.load(f"{HOMEDIR}/revision/Generated_data_revision/video1/Graph_SDI_related/empirical_SDI.npz")['high_beta'])
video_watching_SDI_gamma=np.log2(np.load(f"{HOMEDIR}/revision/Generated_data_revision/video1/Graph_SDI_related/empirical_SDI.npz")['gamma'])
# video_watching_SDI = np.log2(np.load(f"{HOMEDIR}/revision/Generated_data_revision/video1/Graph_SDI_related/empirical_SDI.npz")['wideband'])
for i in range(7):
    idx = np.array(match)==i
    nw_theta.append(np.mean(video_watching_SDI_theta[:, idx], axis=1))
    nw_alpha.append(np.mean(video_watching_SDI_alpha[:, idx], axis=1))
    # nw_wideband.append(np.mean(video_watching_SDI[:, idx], axis=1))
    nw_low_beta.append(np.mean(video_watching_SDI_low_beta[:, idx], axis=1))
    nw_high_beta.append(np.mean(video_watching_SDI_high_beta[:, idx], axis=1))
    nw_gamma.append(np.mean(video_watching_SDI_gamma[:, idx], axis=1))



# Combine the lists into a single DataFrame
df = pd.DataFrame({
    'theta': nw_theta,
    'alpha': nw_alpha,
    'low_beta': nw_low_beta,
    'high_beta': nw_high_beta,
    'gamma': nw_gamma
    # 'wideband': nw_wideband
})

# Get the list of network names
networks = df.columns.tolist()

# Initialize an empty DataFrame to store significance indicators
significance_df = pd.DataFrame(index=df.columns, columns=df.columns)
upper_triangle_values_all_yeo = []
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
    upper_triangle = np.triu(significance_df.astype(float), k=1)
    upper_triangle_values = upper_triangle[np.triu_indices_from(upper_triangle, k=1)]
    upper_triangle_values_all_yeo.append(upper_triangle_values)

    sns.heatmap(significance_df.astype(float), cmap='inferno', annot=True, fmt=".1f", xticklabels=band_labels, yticklabels=band_labels)
    plt.show()

# %%
plt.style.use('fivethirtyeight')
# Convert the list of upper triangle values to a DataFrame
upper_triangle_values_df = pd.DataFrame(upper_triangle_values_all_yeo).T
upper_triangle_values_df.columns = ['Vis', "SomMot", "DorsAttn", "Sal", "Limbic", "FPN", "Default"]

# Melt the DataFrame to long format
upper_triangle_values_long_df = upper_triangle_values_df.melt(var_name='Yeo Network', value_name='Spearman Correlation')

# Create a boxplot with transposed axes
plt.figure(figsize=(12, 6))
sns.boxplot(x='Spearman Correlation', y='Yeo Network', data=upper_triangle_values_long_df)
# %%
import os
os.chdir('/users/local/Venkatesh/structure-function-eeg/src_scripts')
HOMEDIR = "/users/local/Venkatesh/structure-function-eeg/"
import numpy as np
import scipy
import utility_functions
# %%
#rFigure BI_heatmap
band = 'alpha'
condition="video1"
graph = np.load(f"{HOMEDIR}/src_data/individual_graphs.npz")
envelope_signal_bandpassed = np.load(f"{HOMEDIR}/Generated_data/{condition}/cortical_surface_related/{band}_bandpassed.npz")

for sub, signal in envelope_signal_bandpassed.items():
    _, eigenvals, eigenvectors = utility_functions.eigmodes(W = graph[sub])
    lf_comp, hf_comp = utility_functions.fullpipeline(envelope=signal, eigevecs=eigenvectors, eigvals=eigenvals, in_seconds=True)
    hf_comp_reshaped = hf_comp.reshape(360, 170, 125)
    lf_comp_reshaped = lf_comp.reshape(360, 170, 125)
    
    lf_comp_normed = scipy.linalg.norm(lf_comp_reshaped, axis=2)
    hf_comp_normed = scipy.linalg.norm(hf_comp_reshaped, axis=2)
    
    

sns.heatmap(np.log2(np.abs(hf_comp_normed)/np.abs(lf_comp_normed)), cmap='cold_hot', center=0)
# plt.yticks(ticks=np.arange(7)+0.5, labels=['Vis', "SomMot", "DorsAttn", "Sal", "Limbic", "FPN", "DMN"])
plt.xlabel('Time (s)')
plt.ylabel('ROIs')
plt.title('Subject 43, alpha band, Video 1')

# %%
import _6_SDI_statistics
import importlib
importlib.reload(_6_SDI_statistics)
grouplevel_SDI_video1=_6_SDI_statistics.stats_full_test(bands=['theta', 'alpha', 'low_beta', 'high_beta', 'gamma',  'widerband'], condition='video1')

# %%
importlib.reload(_6_SDI_statistics)
grouplevel_SDI_video1_og=_6_SDI_statistics.stats_full_test(bands=['theta', 'alpha', 'low_beta', 'high_beta', 'gamma',  'widerband'], condition='video1')

# %%
upper_triangle_values_all_yeo = []
significance_df = pd.DataFrame(index=['theta', 'alpha', 'low_beta', 'high_beta', 'gamma'], columns=['theta', 'alpha', 'low_beta', 'high_beta', 'gamma'])
for band1 in ['theta', 'alpha', 'low_beta', 'high_beta', 'gamma']:
    for band2 in ['theta', 'alpha', 'low_beta', 'high_beta', 'gamma']:
        if band1 != band2:
            
            stats, p_value = spearmanr(grouplevel_SDI_video1_stft[band1], grouplevel_SDI_video1_stft[band2])
            if p_value > 0.05:
                significance_df.loc[band1, band2] = 0
            else:
                significance_df.loc[band1, band2] = stats

            rho.append(stats)
            p.append(p_value)
# %%
np.nanstd(significance_df.values)
# %%
grouplevel_SDI_video1['widerband']
# %%
from scipy.stats import spearmanr

