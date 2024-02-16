#%%

import numpy as np
import importlib
import utility_functions
importlib.reload(utility_functions)
import _7_SDI_spatial_maps
importlib.reload(_7_SDI_spatial_maps)
from tqdm import tqdm

HOMEDIR = "/users/local/Venkatesh/structure-function-eeg/"
path_Glasser = f"{HOMEDIR}/src_data/Glasser_masker.nii.gz"

def SDI(graph, condition):
    n_surrogates = 19

    empirical_SDI_bundle = dict()
    surrogate_SDI_bundle = dict()

    for band in ['theta', 'alpha', 'low_beta', 'high_beta', 'gamma', 'wideband']:
    
        envelope_signal_bandpassed = np.load(f"{HOMEDIR}/Generated_data/{condition}/cortical_surface_related/{band}_bandpassed.npz")
        s_bundle = list()
        e_bundle = list()

        for sub, signal in tqdm(envelope_signal_bandpassed.items()):
            _, eigenvals, eigenvectors = utility_functions.eigmodes(W = graph[sub])
            _, lf_comp, hf_comp = utility_functions.fullpipeline(envelope=signal, eigevecs=eigenvectors, eigvals=eigenvals)
            empirical_SDI = hf_comp/lf_comp
            e_bundle.append(empirical_SDI)
            surr_signal = utility_functions.surrogate_eigenmodes(eigenvectors, signal, n_surrogates)


            surr_SDI = list()
            for surr in range(np.shape(surr_signal)[0]):
                graph_signal = np.squeeze(surr_signal[surr])

                lf, hf = utility_functions.fullpipeline(graph_signal, eigenvectors, eigenvals, is_surrogate=True)
                SDI = hf/lf
                surr_SDI.append(SDI)

            surr_SDI = np.squeeze(surr_SDI)
            s_bundle.append(surr_SDI)
        
        empirical_SDI_bundle[f'{band}'] = e_bundle
        surrogate_SDI_bundle[f'{band}']= np.swapaxes(s_bundle, 0, 1)
    
    np.savez_compressed(f"{HOMEDIR}/Generated_data/{condition}/Graph_SDI_related/empirical_SDI.npz", **empirical_SDI_bundle)
    np.savez_compressed(f"{HOMEDIR}/Generated_data/{condition}/Graph_SDI_related/surrogate_SDI.npz", **surrogate_SDI_bundle)
#%%
#movie
graph  = np.load(f"{HOMEDIR}/src_data/individual_graphs.npz")
movie_SDI = SDI(graph, 'video1')
#%%
#rest
graph  = np.load(f"{HOMEDIR}/src_data/individual_graphs.npz")
SDI(graph, 'rest')
#%%
#anothervideo
graph  = np.load(f"{HOMEDIR}/src_data/individual_graphs.npz")
SDI(graph, 'video2')
#%%
