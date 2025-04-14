import os
import numpy as np
import mne
from scipy.signal import cheby2, filtfilt
from tqdm import tqdm
import scipy.io as sio
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def getData(subject_index):
    # subject_index: integer, e.g., 3 (1-9)
    
    data_path = '/egr/research-slim/shared/EEG_data/BCICIV/'
    
    # -------------------
    # Process T session
    # -------------------
    session_type = 'T'
    file_T = os.path.join(data_path, f"BCICIV_2a/A0{subject_index}{session_type}.gdf")
    
    # Load the GDF file; preload=True loads data into memory.
    raw_T = mne.io.read_raw_gdf(file_T, preload=True, verbose=False)
    
    # Extract events and select those with code 768.
    events_T, idx_T = mne.events_from_annotations(raw_T, verbose=False)
    selected_events_T = events_T[events_T[:, 2] == idx_T['768']]
    
    # Get the raw data as a NumPy array (shape: n_samples x n_channels)
    s_T = raw_T.get_data().T  # now rows are time points
    n_trials_T = selected_events_T.shape[0]
    data_1 = np.zeros((1000, 22, n_trials_T))
    
    # For each event, extract 1000 samples starting 500 samples after event onset.
    for k, event in enumerate(selected_events_T):
        pos = int(event[0])
        trial_data = s_T[pos + 500 : pos + 1500, :22]  # 1000 samples, first 22 channels
        if trial_data.shape[0] == 1000:
            data_1[:, :, k] = trial_data
 
    # Replace any NaNs with zeros
    data_1[np.isnan(data_1)] = 0

    # Label extraction: depending on your requirements, you might extract labels.
    
    # Step 4: Create a NumPy array of class labels
    label_1 = sio.loadmat(os.path.join(data_path, f"BCICIV_2a_labels/A0{subject_index}{session_type}.mat"))['classlabel']


    # -------------------
    # Process E session
    # -------------------
    session_type = 'E'
    file_E = os.path.join(data_path, f"BCICIV_2a/A0{subject_index}{session_type}.gdf")
    
    raw_E = mne.io.read_raw_gdf(file_E, preload=True, verbose=False)
    events_E, idx_E = mne.events_from_annotations(raw_E, verbose=False)
    selected_events_E = events_E[events_E[:, 2] == idx_E['768']]
    
    s_E = raw_E.get_data().T
    n_trials_E = selected_events_E.shape[0]
    data_2 = np.zeros((1000, 22, n_trials_E))
    
    for k, event in enumerate(selected_events_E):
        pos = int(event[0])
        trial_data = s_E[pos + 500 : pos + 1500, :22]
        if trial_data.shape[0] == 1000:
            data_2[:, :, k] = trial_data
    
    data_2[np.isnan(data_2)] = 0

    # Step 4: Create a NumPy array of class labels
    label_2 = sio.loadmat(os.path.join(data_path, f"BCICIV_2a_labels/A0{subject_index}{session_type}.mat"))['classlabel']
    # -------------------
    # Preprocessing: Band-pass filtering (4-40 Hz)
    # -------------------
    fc = 250  # sampling rate in Hz
    Wl, Wh = 4, 40  # passband frequencies in Hz
    nyq = fc / 2.0
    Wn = [Wl / nyq, Wh / nyq]  # normalized frequencies for filter design
    
    # Design a 6th-order Chebyshev Type II bandpass filter with 60 dB attenuation.
    b, a = cheby2(6, 60, Wn, btype='bandpass')
    
    # Apply zero-phase filtering for each trial (filter along axis=0, which is time).
    for j in range(data_1.shape[2]):
        data_1[:, :, j] = filtfilt(b, a, data_1[:, :, j], axis=0)
    for j in range(data_2.shape[2]):
        data_2[:, :, j] = filtfilt(b, a, data_2[:, :, j], axis=0)

    # -------------------
    # Standardization: subtract mean and divide by standard deviation (across trials)
    # -------------------
    # Compute mean and standard deviation along the third dimension (trials).
    # eeg_mean_1 = np.mean(data_1, axis=2, keepdims=True)
    # eeg_std_1 = np.std(data_1, axis=2, ddof=0, keepdims=True)
    # fb_data_1 = (data_1 - eeg_mean_1) / eeg_std_1

    # eeg_mean_2 = np.mean(data_2, axis=2, keepdims=True)
    # eeg_std_2 = np.std(data_2, axis=2, ddof=0, keepdims=True)
    # fb_data_2 = (data_2 - eeg_mean_2) / eeg_std_2

    # -------------------
    # Save the data in NPZ format
    # -------------------
    save_path = '/egr/research-slim/shared/EEG_data/BCICIV/BCICIV_2a_npz'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Save T session data in NPZ
    save_file_T = os.path.join(save_path, f"A0{subject_index}T.npz")
    np.savez(save_file_T, data=data_1, label=label_1)
    
    # Save E session data in NPZ
    save_file_E = os.path.join(save_path, f"A0{subject_index}E.npz")
    np.savez(save_file_E, data=data_2, label=label_2)
    
    # Optionally, return the processed data
    return None

# Example usage:
if __name__ == '__main__':
    bar = tqdm(range(1,10))
    for i in bar:
        bar.set_description(f"Processing sub{i:02d}")
        getData(i)
    
