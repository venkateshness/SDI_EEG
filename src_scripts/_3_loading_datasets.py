#%%
import mne
import numpy as np
import pandas as pd
import os

HOMEDIR = "/users/local/Venkatesh/structure-function-eeg/"


def csv_to_raw_mne(path_to_file, path_to_montage_ses, fs, path_to_events, state,  montage="GSN-HydroCel-129"):
    
    """Load csv files of data, chan locations and events and return a raw mne instance
    
    Args:
        path_to_file (str): path to the csv file containing the data
        path_to_montage_ses (str): path to the csv file containing the channel locations
        fs (int): sampling frequency
        path_to_events (str): path to the csv file containing the events
        state (str): the state of the subject (rest, video1, video2)
        montage (str): the montage to use for the raw mne instance
        
        
    Returns:
        raw (mne.io.RawArray): raw mne instance
        events_final (np.array): array containing the events"""

    data = np.loadtxt(path_to_file, delimiter=",")
    chans = pd.read_csv(path_to_montage_ses, sep=",", header=None)
    ch_list = [
        "E1",
        "E8",
        "E14",
        "E17",
        "E21",
        "E25",
        "E32",
        "E38",
        "E43",
        "E44",
        "E48",
        "E49",
        "E56",
        "E57",
        "E63",
        "E64",
        "E69",
        "E73",
        "E74",
        "E81",
        "E82",
        "E88",
        "E89",
        "E94",
        "E95",
        "E99",
        "E100",
        "E107",
        "E113",
        "E114",
        "E119",
        "E120",
        "E121",
        "E125",
        "E126",
        "E127",
        "E128",
    ]
    ch_names = list(chans.values[1:, 0]) 

    if state == "Rest": 
        ch_names_appended = list(np.append(ch_names, "stim_channel"))
        types = ["eeg"] * (len(ch_names_appended) - 1)
        types.append("stim")
        data2 = np.zeros([1, len(data[0])])  # len(raw.times)
        data_appended = np.append(data, data2, axis=0)
        info = mne.create_info(ch_names_appended, sfreq=fs, ch_types=types)
        raw = mne.io.RawArray(data_appended, info)

    else:
        types = ["eeg"] * (len(ch_names))
        info = mne.create_info(ch_names, sfreq=fs, ch_types=types)
        raw = mne.io.RawArray(data, info)

    
    raw.set_montage(montage)
    
    if path_to_events:
        # parse events file
        raw_events = pd.read_csv(
            path_to_events, sep=r"\s*,\s*", header=None, engine="python"
        )        
        idx = np.where((raw_events[0]!='break cnt') & (raw_events[0]!='type'))[0]
        samples = raw_events[1][idx].to_numpy()
        event_values = raw_events[0][idx].to_numpy()
            
        events = np.zeros((len(samples), 3))

        events = events.astype("int")
        events[:, 0] = samples
        events[:, 2] = event_values
        
        
        if event_values[-1]=="9999":
            events = events[:-1, :]

        events_final = np.append(events, np.ones((1, 3)), axis=0).astype("int")
        raw = exclude_channels_from_raw(raw, ch_list)

    return raw, events_final


def exclude_channels_from_raw(raw, ch_to_exclude):
    """Return a raw structure where ch_to_exclude are removed
    
    Args:
        raw (mne.io.RawArray): raw mne instance
        ch_to_exclude (list): list of channels to exclude
        
    Returns:
        raw (mne.io.RawArray): raw mne instance with channels removed"""
    idx_keep = mne.pick_channels(
        raw.ch_names, include=raw.ch_names, exclude=ch_to_exclude
    )
    raw.pick_channels([raw.ch_names[pick] for pick in idx_keep])
    return raw


def preparation(filename, state):
    """Load the data and events for a given subject and state
    
    Args:
        filename (str): the subject's name
        state (str): the EEG condition (rest, video1, video2)
        
    Returns:
        raw (mne.io.RawArray): raw mne instance
        events (np.array): array containing the events"""
    
    if state == 'video1':
        path_to_file = f"{HOMEDIR}/src_data/video1/%s/Video3_data.csv" % filename
        path_to_events = f"{HOMEDIR}/src_data/video1/%s/Video3_event.csv" % filename
        path_to_montage_ses = (
            f"{HOMEDIR}/src_data/video1/%s/Video3_chanlocs.csv" % filename
        )
    elif state == 'video2':
        path_to_file = f"{HOMEDIR}/src_data/video2/%s/Video2_data.csv" % filename
        path_to_events = f"{HOMEDIR}/src_data/video2/%s/Video2_event.csv" % filename
        path_to_montage_ses = (
            f"{HOMEDIR}/src_data/video2/%s/Video2_chanlocs.csv" % filename
        )

    fs = 500

    raw, events = csv_to_raw_mne(
        path_to_file,
        path_to_montage_ses,
        fs,
        path_to_events,
        state=state,
        montage="GSN-HydroCel-129",
    )
    return raw, events


def preparation_resting_state(filename, state):
    path_to_file = f"{HOMEDIR}/src_data/rest/%s/RestingState_data.csv" % filename
    path_to_events = (
        f"{HOMEDIR}/src_data/rest/%s/RestingState_event.csv" % filename
    )
    path_to_montage_ses = (
        f"{HOMEDIR}/src_data/rest/%s/RestingState_chanlocs.csv" % filename
    )
    fs = 500

    raw, events = csv_to_raw_mne(
        path_to_file,
        path_to_montage_ses,
        fs,
        path_to_events,
        state=state,
        montage="GSN-HydroCel-129",
    )
    return raw, events

subject_list = sorted(list(os.listdir(f'{HOMEDIR}/src_data/rest')))


"""Restingstate"""

for i in range(1, len(subject_list) + 1):
    if os.path.exists(f"{HOMEDIR}/src_data/video1/{subject_list[i-1]}/") and os.path.exists(f"{HOMEDIR}/src_data/rest/{subject_list[i-1]}/"):
        
        if not os.path.exists(
            f"{HOMEDIR}/revision/Generated_data_revision/rest/preprocessed_dataset/{subject_list[i-1]}"
        ):
            os.makedirs(
                f"{HOMEDIR}/revision/Generated_data_revision/rest/preprocessed_dataset/{subject_list[i-1]}"
            )

        resting_state_raw, resting_state_events = preparation_resting_state(
            subject_list[i - 1], "Rest"
        )
        resting_state_raw.save(
            f"{HOMEDIR}/revision/Generated_data_revision/rest/preprocessed_dataset/{subject_list[i-1]}/raw.fif",
            overwrite=True,
        )
        np.savez_compressed(
            f"{HOMEDIR}/revision/Generated_data_revision/rest/preprocessed_dataset/{subject_list[i-1]}/events.npz",
            resting_state_events=resting_state_events,
        )

# """Video-watching"""

for i in range(1, len(subject_list) + 1):
    
    if os.path.exists(f"{HOMEDIR}/src_data/video1/{subject_list[i-1]}/") and os.path.exists(f"{HOMEDIR}/src_data/rest/{subject_list[i-1]}/"):

        if not os.path.exists(
            f"{HOMEDIR}/revision/Generated_data_revision/video1/preprocessed_dataset/{subject_list[i-1]}"
        ):
            os.makedirs(
                f"{HOMEDIR}/revision/Generated_data_revision/video1/preprocessed_dataset/{subject_list[i-1]}"
            )

        sub_raw, sub_events = preparation(subject_list[i - 1], "video1")
        sub_raw.save(
            f"{HOMEDIR}/revision/Generated_data_revision/video1/preprocessed_dataset/{subject_list[i-1]}/raw.fif",
            overwrite=True,
        )
        np.savez_compressed(
            f"{HOMEDIR}/revision/Generated_data_revision/video1/preprocessed_dataset/{subject_list[i-1]}/events.npz",
            video_watching_events=sub_events,
        )

"""Video-2- Test-Retest"""
for i in range(1, len(subject_list) + 1):
    
    if os.path.exists(f"{HOMEDIR}/src_data/video2/{subject_list[i-1]}/") and os.path.exists(f"{HOMEDIR}/src_data/rest/{subject_list[i-1]}/"):

        if not os.path.exists(
            f"{HOMEDIR}/revision/Generated_data_revision/video2/preprocessed_dataset/{subject_list[i-1]}"
        ):
            os.makedirs(
                f"{HOMEDIR}/revision/Generated_data_revision/video2/preprocessed_dataset/{subject_list[i-1]}"
            )

        sub_raw, sub_events = preparation(subject_list[i - 1], "video2")
        sub_raw.save(
            f"{HOMEDIR}/revision/Generated_data_revision/video2/preprocessed_dataset/{subject_list[i-1]}/raw.fif",
            overwrite=True,
        )
        np.savez_compressed(
            f"{HOMEDIR}revision/Generated_data_revision/video2/preprocessed_dataset/{subject_list[i-1]}/events.npz",
            video_watching_events=sub_events,
        )
# %%
