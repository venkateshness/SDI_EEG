"""Memory intensive jobs below; can spike up to 50GB of memory requirement"""
#%%
import mne
import numpy as np

from scipy.signal import butter, lfilter, hilbert
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
import os.path as op
import os
from nilearn import datasets, surface


HOMEDIR = "/users/local/Venkatesh/structure-function-eeg" 
# A very nice overview of the Source Localization workflow : https://mne.tools/stable/overview/cookbook.html
#%%
with np.load(
    f"{HOMEDIR}/src_data/sourcespace_to_glasser_labels.npz"
) as dobj:  # shoutout to https://github.com/rcruces/2020_NMA_surface-plot.git
    atlas = dict(**dobj)


def averaging_by_parcellation(sub):
    """Aggregating the native brain surface fsaverage vertices to parcels (180 each hemi)

    Args:
        sub (array): source time course per subject

    Returns:
        source_signal_in_parcels : signal in parcels
    """
    source_signal_in_parcels = list()
    for roi in list(set(atlas["labels_R"]))[:-1]:
        source_signal_in_parcels.append(
            np.mean(sub.rh_data[np.where(roi == atlas["labels_R"])], axis=0)
        )

    for roi in list(set(atlas["labels_L"]))[:-1]:
        source_signal_in_parcels.append(
            np.mean(sub.lh_data[np.where(roi == atlas["labels_L"])], axis=0)
        )

    return source_signal_in_parcels


def epochs_slicing(
    subject_raw, subject_events, event_list, tmin, tmax, fs, epochs_to_slice
):
    """Slicing only the epochs are in need

    Args:
        subject_raw (array): the signal containing different events
        subject_events (array): event timestamp at onset
        epochs_list (array): event labels
        tmin (int): onset
        tmax (int): onset + max time needed
        fs (int): sampling frequency
        epochs_to_slice (string): the event label

    Returns:
        _type_: _description_
    """

    epochs = mne.Epochs(
        subject_raw,
        subject_events,
        event_list,
        tmin=tmin,
        tmax=tmax,
        preload=True,
        verbose=False,
        baseline=(0, None),
    )
    epochs_resampled = epochs.resample(fs, verbose=False)  # Downsampling

    return epochs_resampled[epochs_to_slice]


# Data-loading
def noise_covariance(subject):
    """Computing Noise Covariance on the EEG signal. Oversimplifying, it is to set what level the noise is in the system

    Args:
        subject (string): Subject ID

    Returns:
        covariance : the computed noise covariance matrix
    """
    raw_resting_state, events_resting_state = (
        mne.io.read_raw_fif(
            f"{HOMEDIR}/Generated_data/rest/preprocessed_dataset/{subject}/raw.fif",
            verbose=False,
        ),
        np.load(
            f"{HOMEDIR}/Generated_data/rest/preprocessed_dataset/{subject}/events.npz"
        )["resting_state_events"],
    )

    epochs = mne.Epochs(
        raw_resting_state,
        events_resting_state,
        [20, 30, 90],
        tmin=0,
        tmax=20,
        preload=True,
        baseline=(0, None),
        verbose=False,
    )
    downsampled_fs = 250
    epochs_resampled = epochs.resample(downsampled_fs)  # Downsampling to 250Hz

    np.random.seed(55)

    # Destroying temporality
    rand = np.random.randint(1, downsampled_fs * 20, size=500)  # 20s
    cov = mne.EpochsArray(
        epochs_resampled["20"][0].get_data()[:, :, rand],
        info=raw_resting_state.info,
        verbose=False,
    )  # event '20' = RS eyes open
    cov.set_eeg_reference("average", projection=True)
    cov.apply_proj()
    covariance = mne.compute_covariance(cov, method="auto", verbose=False)

    return covariance


def forward_model(raw, trans, source_space, bem):
    """Forward solution; roughly modeling various compartements between scalp and cortical mesh

    Args:
        raw (raw.info): the raw data structure from MNE
        trans (string):
        source_space (fsaverage model): Freesurfer Cortical Mesh; fsaverage5 containing 10242 per hemisphere
        bem (bem model): Modeling the electromagnetic conductivity of various compartments
        such as scalp, skull, cortical space

    Returns:
        fwd_model: the forward solution
    """
    fwd_model = mne.make_forward_solution(
        raw.info, trans=trans, src=source_space, bem=bem, eeg=True, verbose=False
    )

    return fwd_model


def making_inverse_operator(raw, fwd_model, subject):

    covariance = noise_covariance(subject)
    inverse_operator = make_inverse_operator(
        raw.info, fwd_model, covariance, verbose=False
    )

    return inverse_operator


def source_locating(epochs, inverse_operator):
    method = "eLORETA"
    snr = 3.0
    lambda2 = 1.0 / snr ** 2
    stcs = apply_inverse_epochs(
        epochs,
        inverse_operator,
        lambda2=lambda2,
        method=method,
        verbose=False,
        return_generator=False,
    )
    return stcs


fs_dir = fetch_fsaverage(subjects_dir=None, verbose=True)
subjects_dir = op.dirname(fs_dir)

fs = 125
n_sub = 43

subject = "fsaverage"
trans = "fsaverage"
source_space = op.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
bem = op.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")

subjects = sorted(list(os.listdir(f'{HOMEDIR}/Generated_data/video1/preprocessed_dataset/')))

n_chans_after_preprocessing = 91
time_in_samples = 21250
stc_bundle = dict()
eloreta_activation = list()

parcellated = dict()
eloreta_activation_variance = dict()
envelope_subs = dict()


for id in range(len(subjects)): 

    data_video, events_list = (
        mne.io.read_raw_fif(
            f"{HOMEDIR}/Generated_data/video1/preprocessed_dataset/{subjects[id]}/raw.fif",
            verbose=False,
        ),
        np.load(
            f"{HOMEDIR}/Generated_data/video1/preprocessed_dataset/{subjects[id]}/events.npz"
        )["video_watching_events"],
    )

    sliced_epoch = epochs_slicing(
        data_video,
        events_list,
        [83, 103, 9999],
        tmin=0,
        tmax=170,
        fs=500,
        epochs_to_slice="83",
    )

    info_d = mne.create_info(
        data_video.info["ch_names"], sfreq=125, ch_types="eeg", verbose=False
    )

    the_epoch = mne.EpochsArray(
        sliced_epoch,
        mne.create_info(
            data_video.info["ch_names"], sfreq=500, ch_types="eeg", verbose=False
        ),
    ).resample(125)
    
    the_epoch.set_eeg_reference("average", projection=True)
    the_epoch.apply_proj()

    raw = mne.io.RawArray(
        the_epoch.get_data().reshape(n_chans_after_preprocessing, time_in_samples),
        info_d,
        verbose=False,
    )


    if id == 0:  # Reusing the fwd_model; cut down some time
        print("Forward model running")
        fwd_model = forward_model(
            data_video, trans=trans, source_space=source_space, bem=bem
        )



    print("Inverse model running....")
    inverse_operator = making_inverse_operator(raw, fwd_model, subjects[id])
    eloreta_activation = source_locating(the_epoch, inverse_operator)


    parcellated[subjects[id]] = averaging_by_parcellation(eloreta_activation[0])

    print('Done with subject', id)
    del raw, the_epoch, data_video, events_list, sliced_epoch, info_d, eloreta_activation, inverse_operator

"""Now that the source localization has been performed and is in the fsaverage native space of having 20k vertices,
it is time to apply Glasser et al. 2016 parcellation on top"""

np.savez_compressed(f"{HOMEDIR}/Generated_data/video1/cortical_surface_related/parcellated_widerband.npz", **parcellated)

video_watching_bundle_STC = parcellated

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Define band ranges
band_ranges = {
    'alpha': (8, 13),
    'low_beta': (13, 20),
    'high_beta': (20, 30),
    'gamma': (30, 40),
    'theta': (4, 8),
    'wideband': (4, 40)
}

# Create a dictionary to store band data for each subject
bands = {}

# Apply bandpass filter and store for each band and each subject
for band, (low, high) in band_ranges.items():
    band_data = {}
    for sub_id, data in video_watching_bundle_STC.items():
        bandpassed = butter_bandpass_filter(data, lowcut=low, highcut=high, fs=125)
        hilberted = hilbert(bandpassed, N=None, axis=-1)
        band_data[f'{sub_id}'] = np.abs(hilberted)
    
    bands[band] = band_data
    np.savez_compressed(f'{HOMEDIR}/Generated_data/video1/cortical_surface_related/{band}_bandpassed', **band_data)
