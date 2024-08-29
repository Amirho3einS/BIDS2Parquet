from scipy import signal
import numpy as np

import mne
import pyedflib


def find_key_by_value(dictionary, target_value):
    keys = [key for key, value in dictionary.items() if value == target_value]
    assert (
        len(keys) == 1
    ), f"Expected to find exactly one key with value '{target_value}', but found {len(keys)} keys."
    return keys[0]


def pre_process_ch(ch_data, fs):
    """Pre-process EEG data by applying a 0.5 Hz highpass
    filter, a 60  Hz lowpass filter and a 50 Hz notch filter,
    all 4th order Butterworth filters. The data is resampled to
    200 Hz.

    Args:
        ch_data: a list or numpy array containing the data of
            an EEG channel
        fs: the sampling frequency of the data

    Returns:
        ch_data: a numpy array containing the processed EEG data
        fs_resamp: the sampling frequency of the processed EEG data
    """
    # Checking the n_channels are the second axis
    assert (
        ch_data.shape[1] < ch_data.shape[0]
    ), "The data is not in the expected shape. The number of channels should be the second axis."
    b, a = signal.butter(4, 0.5 / (fs / 2), "high")
    ch_data = signal.filtfilt(b, a, ch_data, axis=0)

    b, a = signal.butter(4, 60 / (fs / 2), "low")
    ch_data = signal.filtfilt(b, a, ch_data, axis=0)

    b, a = signal.butter(4, [49.5 / (fs / 2), 50.5 / (fs / 2)], "bandstop")
    ch_data = signal.filtfilt(b, a, ch_data, axis=0)

    return ch_data


def mne_edf_data(edf_dir):
    """
    Read and extract data from an EDF file using MNE library.
    Note: The range is on Volts so we have e-6 every where.

    Parameters:
        edf_dir (str): The directory path of the EDF file.

    Returns:
        numpy.ndarray: The extracted EDF data in the shape (Datalen, n_channels).
    """
    data_file = mne.io.read_raw_edf(edf_dir, verbose="error")
    edf_data = data_file.get_data().T  # Datalen, n_channels
    return edf_data


def pyedf_edf_data(edf_dir, n_channles):
    """
    Read EDF data from the specified directory using pyedflib.
    Note: The range is on  micro Volts so abs(values) are > 1.

    Args:
        edf_dir (str): The directory path of the EDF file.
        n_channles (int): The number of channels in the EDF file.

    Returns:
        numpy.ndarray: The EDF data as a numpy array of shape (data_len, n_channels).
    """
    with pyedflib.EdfReader(edf_dir) as f:
        data = [f.readSignal(i) for i in range(n_channles)]
        edf_data = np.stack(data, axis=-1)  # (data_len, n_channels)
    return edf_data
