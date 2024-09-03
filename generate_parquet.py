#%%
%load_ext autoreload
%autoreload 2

#%%
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime, timedelta
from typing import TypedDict

from epilepsy2bids.annotations import Annotations
from bids import BIDSLayout

from src.utils import find_key_by_value, pre_process_ch, mne_edf_data, pyedf_edf_data
# %%
class Segment(TypedDict):
    start: (
        int  # start time of the event from the beginning of the recording, in seconds
    )
    end: int  # end time of the event, in seconds 
    label: int  # lable of the event can be translated with label_SZtype_dict
    bids_name: str  # the file name in the BIDS format
    time_stamp: str  # start time of the event datetime.strftime('%Y-%m-%dT%H:%M:%S')

#%%
dataset_dir = "/Users/amir/PhD/seizure/BIDS2Parq/chbmit"
layout = BIDSLayout(dataset_dir)
# %%
DATASET = 'CHBMIT'
assert DATASET=='CHBMIT', "Dataset is not CHBMIT"
# %%

# %%
events_dir = os.path.join(dataset_dir, "events.json")
with open(events_dir, "r") as f:
    events = json.load(f)
    sz_types = events["Levels"]
# %%
label_SZtype_dict = {index: key for index, (key, value) in enumerate(sz_types.items())}
# %%
# Creating Parquet folder structure
PARQUET_DIR = "/Users/amir/PhD/seizure/BIDS2Parq/parquet"
if not os.path.exists(PARQUET_DIR):
    os.makedirs(PARQUET_DIR)
parquet_dataset_dir = os.path.join(PARQUET_DIR, DATASET)
if not os.path.exists(parquet_dataset_dir):
    os.makedirs(parquet_dataset_dir)

# %%
# Adjusted list of subjects as a range from 1 to 25 (formatted with leading zeros)
# CHBMIT has 24 subjects, but the subject '21' is actually session 2 of subject '01'
subjects = [f'{i:02}' for i in range(1, 25)]

for subject in subjects:
    print(f"Processing subject {subject}...")
    # CHBMIT - Handle special case where subject '21' is actually session 2 of subject '01'
    if subject == '21':
        actual_subject = '01'
        session = '02'
    else:
        actual_subject = subject
        session = '01' # CHBMIT has only one session for each subject

    # Get all files for this subject
    files = layout.get(subject=actual_subject, session=session, extension=['.edf', '.json', '.tsv'])

    # Group the files by their common entities (e.g., task, run)
    grouped_files = {}
    for f in files:
        key = (f.entities['task'], f.entities.get('run'))
        if key not in grouped_files:
            grouped_files[key] = []
        grouped_files[key].append(f)
        
    # Convert the keys to a sorted list based on the run number
    sorted_keys = sorted(grouped_files.keys(), key=lambda x: int(x[1]))

    # Check if the keys are in the correct order (based on the run number)
    original_keys = list(grouped_files.keys())
    assert original_keys == sorted_keys, f"The files for subject {actual_subject} and session {session} are not in the expected order."

    # # DEBUG Print the grouped files
    # # Now, `grouped_files` dictionary has the matched files
    # for key, matched_files in grouped_files.items():
    #     print(f"Matched files for {actual_subject} - {key}:")
    #     for mf in matched_files:
    #         print(mf.path)

    df_list = []
    duration = 0
    global_index = 0
    segments = []
    # Creating Huge DataFrame for the subject
    for group in grouped_files:
        # print(group)
        edf_dir = grouped_files[group][0].path
        info_dir = grouped_files[group][1].path
        tsv_dir = grouped_files[group][2]
        with open(info_dir) as f:
            info = json.load(f)
        fs = info["SamplingFrequency"]
        n_channles = info["EEGChannelCount"]
        duration += info["RecordingDuration"]
        edf_data = mne_edf_data(edf_dir)    # data_len (all_rec), n_channels
        edf_data = pre_process_ch(edf_data, fs)
        annot = Annotations.loadTsv(tsv_dir)

        data_len = int(annot.events[0]["recordingDuration"])
        # Obtain the data
        # dataTime is shared for all events
        start_time = annot.events[0]["dateTime"]   
        reshaped_data = edf_data[:data_len* fs, :].reshape(data_len, fs, n_channles)

        # Obtain the labels
        # labels = np.zeros(data_len)

        # Checking the file does not strat with a seizure
        if annot.events[0]["onset"] == 0 and annot.events[0]["eventType"].value != label_SZtype_dict[0]:
            print("WARNING: file starts with a seizure")
        

        if annot.events[0]["eventType"].value != label_SZtype_dict[0]:
            # Adding First non-seizure segment
            start = global_index
            end = start + int(annot.events[0]["onset"])
            label = 0
            time_stamp = start_time
            segment = {
                "start": start,
                "end": end,
                "label": label,
                "bids_name": edf_dir[:-4].split("/")[-1],
                "time_stamp": time_stamp.strftime('%Y-%m-%dT%H:%M:%S')
            }
            segments.append(segment)
            for event in annot.events:
                assert event["onset"] == int(event["onset"]), f"FLOAT ONSET: {event['onset']} Subject: {subject}, File: {edf_dir[:-4]}"
                assert event["duration"] == int(event["duration"]), f"FLOAT DURATION: {event['duration']} Subject: {subject}, File: {edf_dir[:-4]}"
                
                start = int(event["onset"]) + global_index
                end = start + int(event["duration"])
                label = find_key_by_value(label_SZtype_dict, event["eventType"].value)
                time_stamp = start_time + timedelta(seconds=start - global_index)
                # if event["eventType"] != label_SZtype_dict[0]:
                #     labels[int(event["onset"]):int(event["onset"] + event["duration"])] = find_key_by_value(label_SZtype_dict, event["eventType"].value)
                # Create a segment
                segment = {
                    "start": start,
                    "end": end,
                    "label": label,
                    "bids_name": edf_dir[:-4].split("/")[-1],
                    "time_stamp": time_stamp.strftime('%Y-%m-%dT%H:%M:%S')
                }
                segments.append(segment)
            
            # add the last non-seizure segment
            start = end
            end = global_index + data_len
            label = 0
            time_stamp = start_time + timedelta(seconds=start - global_index)
            segment = {
                "start": start,
                "end": end,
                "label": label,
                "bids_name": edf_dir[:-4].split("/")[-1],
                "time_stamp": time_stamp.strftime('%Y-%m-%dT%H:%M:%S')
            }
            segments.append(segment)
        else:      # In case we only have non-seizure events
            start = global_index
            end = start + data_len
            label = 0
            time_stamp = start_time
            segment = {
                "start": start,
                "end": end,
                "label": label,
                "bids_name": edf_dir[:-4].split("/")[-1],
                "time_stamp": time_stamp.strftime('%Y-%m-%dT%H:%M:%S') 
            }
            segments.append(segment)
        global_index = end
        # Obtaining the timestamps
        # timestamps = [start_time + timedelta(seconds=i) for i in range(data_len)]
        # unix_timestamps = [t.timestamp() for t in timestamps]
        # Construct the DataFrame
        df_list.append(pd.DataFrame({
            'data': list(reshaped_data),  # Convert to list for DataFrame
            # 'label': labels,
            # 'timestamp': unix_timestamps
        }))
    # Concatenate all the DataFrames
    df_all = pd.concat(df_list, ignore_index=True)

    # df flattening
    df_all['data'] = df_all['data'].apply(lambda x: x.flatten())
    df_all.attrs['subject'] = subject
    df_all.attrs['szTypes'] = sz_types
    df_all.attrs['label_SZtype_dict'] = label_SZtype_dict
    info['RecordingDuration'] = duration
    df_all.attrs['info'] = info
    df_all.attrs['data_len'] = data_len
    df_all.attrs['data_shape'] = (fs, n_channles)
    df_all.attrs['segments'] = segments

    # Creating parquet folder and file
    parquet_subject_dir = os.path.join(parquet_dataset_dir, subject)
    if not os.path.exists(parquet_subject_dir):
        os.makedirs(parquet_subject_dir)
    parquet_file = os.path.join(parquet_subject_dir, f"{subject}.parquet")
    df_all.to_parquet(parquet_file,
                      index=False,
                      compression='lz4',
                      row_group_size=512)
    meta_data_file = os.path.join(parquet_subject_dir, f"{subject}_meta.json")
    # Save the attributes separately
    with open(meta_data_file, "w") as f:
        json.dump(df_all.attrs, f)
# %%
# Testing Parquet file
    
df_test = pd.read_parquet(parquet_file)
# %%
data_test = df_test['data']
# %%
data_test[0].shape
# %%
# Spent some time raises an error, going with parquet
# from copy import deepcopy
# df_cpy = deepcopy(df_all)
# df_cpy['data'] = df_cpy['data'].apply(lambda x: np.array(x, dtype=np.float32))
#%%

# hdf5_dir = "/Users/amir/PhD/seizure/BIDS2Parq/parquet/CHBMIT/08.hdf5_gzip"

# df_dict = {
#     'data': np.stack(df_cpy['data'].values),  # Stack into a 2D NumPy array
#     'label': df_cpy['label'].values,
#     'timestamp': df_cpy['timestamp'].values
# }

# # Convert the dictionary back to a DataFrame
# df_fixed = pd.DataFrame(df_dict)

# df_cpy.to_hdf(
#                     hdf5_dir,
#                     key="eeg",
#                     mode="w",
#                     format="table",
#                     complib="lzo",
#                     complevel=9,
#                 )
# %%
# hdf5_dir = "/Users/amir/PhD/seizure/BIDS2Parq/parquet/CHBMIT/08.hdf5_gzip"

# df_all.to_hdf(
#                     hdf5_dir,
#                     key="eeg",
#                     mode="w",
#                     format="table",
#                     complib="lzo",
#                     complevel=9,
#                 )
# %%
