#%%


import numpy as np
import importlib
import utility_functions
importlib.reload(utility_functions)
import _7_SDI_spatial_maps
importlib.reload(_7_SDI_spatial_maps)
from tqdm import tqdm

HOMEDIR = "/users/local/Venkatesh/structure-function-eeg/"
graph  = np.load(f"{HOMEDIR}/src_data/individual_graphs.npz")

condition = 'rest'
for band in ['widerband']:
    
    if not band == 'widerband':
        
        envelope_signal_bandpassed = np.load(f"{HOMEDIR}/Generated_data/{condition}/cortical_surface_related/{band}_bandpassed.npz")
    else:
        envelope_signal_bandpassed = np.load(f"{HOMEDIR}/Generated_data/{condition}/cortical_surface_related/parcellated_widerband.npz")

critical_freq_subjects = []
for sub, signal in tqdm(envelope_signal_bandpassed.items()):
    _, eigenvals, eigenvectors = utility_functions.eigmodes(W = graph[sub])
    
    gpsd_averaged, _ = utility_functions.compute_gpsd(signal, eigenvectors)
    critical_freq = utility_functions.split_gpsd(gpsd_averaged, eigenvals)
    critical_freq_subjects.append(critical_freq)
# %%
print("Rest, Median:")
np.median(critical_freq_subjects)

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

# Generate a sample signal: composite sinusoidal wave
fs = 125  # Sampling frequency
t = np.linspace(0, 1, fs, endpoint=False)  # Time axis
frequencies = [8, 10, 15]  # Frequencies in Hz
signal = sum(np.sin(2 * np.pi * f * t) for f in frequencies)  # Sum of sinusoids

# Apply Short Time Fourier Transform
f, t, Zxx = stft(signal, fs=fs, nperseg=37)

# Plot the STFT magnitude
plt.figure(figsize=(10, 6))
plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(label='Magnitude')
plt.show()

# %%
f
# %%
t
# %%
37//2
# %%
