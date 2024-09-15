from nilearn.regions import signals_to_img_labels
from nilearn.datasets import fetch_icbm152_2009
from nilearn import plotting

import numpy as np
from nilearn import plotting
import matplotlib.pyplot as plt
import itertools
from nilearn._utils import check_niimg_3d
from matplotlib import gridspec
from nilearn.surface import vol_to_surf
import numpy.linalg as la
import tqdm as tqdm
import scipy.io as sio
    

regions = 360
def eigmodes(W):
    """returns normalized laplacian from weight matrix W.
    thr is a proportional weight threshold
    Returns:
        Matrix of floats: A weight matrix for the thresholded graph
    """

    degree = np.diag(np.power(np.sum(W, axis=1), -0.5))
    laplacian = np.eye(W.shape[0]) - np.matmul(
        degree, np.matmul(W, degree)
    )
    [eigvals, eigevecs] = la.eigh(laplacian)
    
    return laplacian, eigvals, eigevecs


def surrogate_eigenmodes(eigvector, signal, n_surrogate):  # updated
    """Surrogate data generation
    Args:
        eigvector (array): Eigenvector
        signal (array): Cortical brain/graph signal
    Returns:
        surrogate_signal (array): Surrogate signal"""
    
    surrogate_signal = list()
    for n in range(n_surrogate):

        np.random.seed(n)
        random_signs = np.round(
            np.random.rand(
                regions,
            )
        )
        random_signs[random_signs == 0] = -1
        random_signs = np.diag(random_signs)

        g_psd = np.matmul(eigvector.T, signal)
        eigvector_manip = np.matmul(eigvector, random_signs)
        reconstructed_signal = np.matmul(eigvector_manip, g_psd)
        surrogate_signal.append(reconstructed_signal)

    # assert np.shape(surrogate_signal) == (
    #     n_surrogate,
    #     regions,
    #     21250
    # )

    return surrogate_signal


def compute_gpsd(signal,eigenvecs):

    gft = np.matmul(eigenvecs.T, signal)
    psd_abs_squared = np.sqrt(np.power(np.abs(gft), 2))
    gpsd_averaged = np.mean(psd_abs_squared, axis=1)
    

    return gpsd_averaged, psd_abs_squared

def split_gpsd(gpsd,eigenvals):

    halfpower = np.trapz(x=eigenvals,y=gpsd) / 2
    
    sum_of_freqs = 0
    i = 0
    while sum_of_freqs < halfpower:
        sum_of_freqs = np.trapz(x=eigenvals[:i],y=gpsd[:i])
        i += 1
    
    critical_freq = i - 1
    return critical_freq
    

def fullpipeline(envelope, eigevecs, eigvals, is_surrogate=False, in_seconds=False):
    signal_for_gft = envelope

    
    psd = np.matmul(np.array(eigevecs).T, signal_for_gft)

    psd_power_avg, psd_power = compute_gpsd(signal_for_gft, eigevecs)
    critical_freq = split_gpsd(psd_power_avg, eigvals)
    low_freq =  np.zeros((regions, regions))
    low_freq[:,:critical_freq] = np.array(eigevecs)[ :, :critical_freq]

    high_freq =  np.zeros((regions, regions))
    high_freq[:,critical_freq:] = np.array(eigevecs)[ :, critical_freq:]

    low_freq_component = np.matmul(low_freq, psd)
    high_freq_component = np.matmul(high_freq, psd)

    low_freq_component_reshaped_normed = np.linalg.norm(low_freq_component, axis=-1)
    high_freq_component_reshaped_normed = np.linalg.norm(high_freq_component, axis=-1)
    
    if is_surrogate:
       return low_freq_component_reshaped_normed, high_freq_component_reshaped_normed
    
    if not is_surrogate:
        if not in_seconds:
            SDI = np.log2(high_freq_component_reshaped_normed / low_freq_component_reshaped_normed)
            return SDI, low_freq_component_reshaped_normed, high_freq_component_reshaped_normed
        elif in_seconds:
            return low_freq_component, high_freq_component


def fullpipeline_time_varying_SDI(envelope, eigevecs, eigvals, is_surrogate=False, in_seconds=False):
    
    signal_for_gft = envelope
    
    psd_reshaped = np.matmul(np.array(eigevecs).T, signal_for_gft)
    seconds = int(np.shape(signal_for_gft)[1]/125)
    psd_power_avg, psd_power = compute_gpsd(signal_for_gft, eigevecs)
    psd_power_time_avg = np.mean(np.reshape(psd_power, (regions, seconds, 125)), axis=-1)
    
    critical_freq = [split_gpsd(psd_power_time_avg[:, i], eigvals) for i in range(np.shape(psd_power_time_avg)[1])]

    low_freq_component_all = list()
    high_freq_component_all = list()
    
    for id, critical_f in enumerate(critical_freq):
        
        low_freq =  np.zeros((regions, regions))
        low_freq[:,:critical_f] = np.array(eigevecs)[ :, :critical_f]

        high_freq =  np.zeros((regions, regions))
        high_freq[:,critical_f:] = np.array(eigevecs)[ :, critical_f:]
        
        low_freq_component = np.matmul(low_freq, psd_reshaped[:, id*125: id*125+125])
        high_freq_component = np.matmul(high_freq, psd_reshaped[:, id*125: id*125+125])

        low_freq_component_all.append(low_freq_component)
        high_freq_component_all.append(high_freq_component)

    low_freq_component_reshaped_normed = np.linalg.norm(low_freq_component_all, axis=(2))
    high_freq_component_reshaped_normed = np.linalg.norm(high_freq_component_all, axis=(2))
    
    
    return low_freq_component_reshaped_normed, high_freq_component_reshaped_normed
    



def surrogate_eigenmodes_uninformed(eigvector, signal, n_surrogate):  # updated
    """Surrogate data generation
    Args:
        eigvector (array): Eigenvector
        signal (array): Cortical brain/graph signal
    Returns:
        surrogate_signal (array): Surrogate signal"""
    
    surrogate_signal = list()
    for n in range(n_surrogate):

        np.random.seed(n)
        random_signs = np.round(
            np.random.rand(
                regions,
            )
        )
        random_signs[random_signs == 0] = -1
        random_signs = np.diag(random_signs)


        g_psd = np.matmul(eigvector.T, signal)
        eigvector = np.fliplr(eigvector)
        eigvector_manip = np.matmul(eigvector, random_signs)
        reconstructed_signal = np.matmul(eigvector_manip, g_psd)
        surrogate_signal.append(reconstructed_signal)

    assert np.shape(surrogate_signal) == (
        n_surrogate,
        regions,
    )

    return surrogate_signal