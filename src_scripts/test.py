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
f, t, Zxx = stft(signal, fs=fs, nperseg=25)

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
from scipy.signal import stft, istft
import numpy as np
band_ranges = {
    'alpha': (8, 13),
    'low_beta': (13, 20),
    'high_beta': (20, 30),
    'gamma': (30, 40),
    'theta': (4, 8)
    }

fs=125
data = np.random.random((10,1250))
# Apply Short Time Fourier Transform
f, t, Zxx = stft(signal, fs=fs, nperseg=25)

# Plot the STFT magnitude
plt.figure(figsize=(10, 6))
plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(label='Magnitude')
plt.show()

# %%
plt.plot(np.abs(Zxx)[3])
# %%
frequencies
# %%
import numpy as np
import matplotlib.pyplot as plt

# Parameters
fs = 125  # Sampling frequency (Hz)
t = np.arange(0, 1.0, 1/fs)  # Time vector (1 second duration)
freqs = [50, 200, 350]  # Frequencies in Hz
amplitudes = [1, 0.5, 0.2]  # Amplitude of each frequency component

# Generate signal
signal = np.zeros_like(t)
for freq, amplitude in zip(freqs, amplitudes):
    signal += amplitude * np.sin(2 * np.pi * freq * t)

# Add noise
noise = 0.05 * np.random.randn(len(t))
signal += noise

# Plot the signal
plt.figure(figsize=(10, 4))
plt.plot(t, signal)
plt.title('Synthetic Time Series Data')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.show()

from scipy.signal import stft

# STFT parameters
nperseg = 125  # Length of each segment for STFT

# Compute STFT
frequencies, times, Zxx = stft(signal, fs=fs, nperseg=nperseg)
Zxx_magnitude = np.abs(Zxx)

# Plot the STFT result
plt.figure(figsize=(10, 4))
plt.pcolormesh(times, frequencies, Zxx_magnitude, shading='gouraud')
plt.title('STFT Magnitude')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (seconds)')
plt.colorbar(label='Magnitude')

plt.show()

# %%
np.abs(Zxx)
# %%

# %%

# %%
frequencies
# %%
