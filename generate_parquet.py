import os
import json
import argparse
from datetime import datetime, timedelta
from typing import TypedDict
import numpy as np
import pandas as pd
from epilepsy2bids.annotations import Annotations
from bids import BIDSLayout
from src.utils import (
    find_key_by_value,
    pre_process_ch,
    mne_edf_data,
)


# Define a TypedDict for segment metadata
class Segment(TypedDict):
    start: int  # Start time of the event in seconds
    end: int  # End time of the event in seconds
    label: int  # Label of the event (e.g., seizure type)
    bids_name: str  # File name in BIDS format
    time_stamp: str  # Timestamp of the event in ISO format


def process_subject(
    subject, layout, label_SZtype_dict, sz_types, parquet_dataset_dir
):
    """
    Process a single subject's data, convert it to Parquet format, and save metadata.

    Args:
        subject (str): Subject ID.
        layout (BIDSLayout): BIDS layout object for accessing dataset files.
        label_SZtype_dict (dict): Mapping of seizure type labels.
        sz_types (dict): Seizure types from the events.json file.
        parquet_dataset_dir (str): Directory to save the Parquet files.
    """
    print(f"Processing subject {subject}...")

    # Handle special case for subject '21' (session 2 of subject '01')
    if subject == "21":
        actual_subject = "01"
        session = "02"
    else:
        actual_subject = subject
        session = "01"

    # Retrieve all files for the subject
    files = layout.get(
        subject=actual_subject,
        session=session,
        extension=[".edf", ".json", ".tsv"],
    )

    # Group files by common entities (e.g., task, run)
    grouped_files = {}
    for f in files:
        key = (f.entities["task"], f.entities.get("run"))
        if key not in grouped_files:
            grouped_files[key] = []
        grouped_files[key].append(f)

    # Ensure files are sorted by run number
    sorted_keys = sorted(grouped_files.keys(), key=lambda x: int(x[1]))
    original_keys = list(grouped_files.keys())
    assert (
        original_keys == sorted_keys
    ), f"Files for subject {actual_subject} and session {session} are not in the expected order."

    df_list = []
    duration = 0
    global_index = 0
    segments = []

    # Process each group of files
    for group in grouped_files:
        edf_dir = grouped_files[group][0].path
        info_dir = grouped_files[group][1].path
        tsv_dir = grouped_files[group][2]

        # Load metadata and annotations
        with open(info_dir) as f:
            info = json.load(f)
        fs = info["SamplingFrequency"]
        n_channels = info["EEGChannelCount"]
        duration += info["RecordingDuration"]
        edf_data = mne_edf_data(edf_dir)
        edf_data = pre_process_ch(edf_data, fs)
        annot = Annotations.loadTsv(tsv_dir)

        data_len = int(annot.events[0]["recordingDuration"])
        start_time = annot.events[0]["dateTime"]
        reshaped_data = edf_data[: data_len * fs, :].reshape(
            data_len, fs, n_channels
        )

        # Check if the file starts with a seizure
        if (
            annot.events[0]["onset"] == 0
            and annot.events[0]["eventType"].value != label_SZtype_dict[0]
        ):
            print("WARNING: file starts with a seizure")

        # Process events and create segments
        if annot.events[0]["eventType"].value != label_SZtype_dict[0]:
            start = global_index
            end = start + int(annot.events[0]["onset"])
            label = 0
            time_stamp = start_time
            segment = {
                "start": start,
                "end": end,
                "label": label,
                "bids_name": edf_dir[:-4].split("/")[-1],
                "time_stamp": time_stamp.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            segments.append(segment)

            for event in annot.events:
                assert event["onset"] == int(
                    event["onset"]
                ), f"FLOAT ONSET: {event['onset']} Subject: {subject}, File: {edf_dir[:-4]}"
                assert event["duration"] == int(
                    event["duration"]
                ), f"FLOAT DURATION: {event['duration']} Subject: {subject}, File: {edf_dir[:-4]}"

                start = int(event["onset"]) + global_index
                end = start + int(event["duration"])
                label = find_key_by_value(
                    label_SZtype_dict, event["eventType"].value
                )
                time_stamp = start_time + timedelta(
                    seconds=start - global_index
                )
                segment = {
                    "start": start,
                    "end": end,
                    "label": label,
                    "bids_name": edf_dir[:-4].split("/")[-1],
                    "time_stamp": time_stamp.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                segments.append(segment)

            start = end
            end = global_index + data_len
            label = 0
            time_stamp = start_time + timedelta(seconds=start - global_index)
            segment = {
                "start": start,
                "end": end,
                "label": label,
                "bids_name": edf_dir[:-4].split("/")[-1],
                "time_stamp": time_stamp.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            segments.append(segment)
        else:
            start = global_index
            end = start + data_len
            label = 0
            time_stamp = start_time
            segment = {
                "start": start,
                "end": end,
                "label": label,
                "bids_name": edf_dir[:-4].split("/")[-1],
                "time_stamp": time_stamp.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            segments.append(segment)

        global_index = end
        df_list.append(pd.DataFrame({"data": list(reshaped_data)}))

    # Combine all data into a single DataFrame
    df_all = pd.concat(df_list, ignore_index=True)
    df_all["data"] = df_all["data"].apply(lambda x: x.flatten())
    df_all.attrs["subject"] = subject
    df_all.attrs["szTypes"] = sz_types
    df_all.attrs["label_SZtype_dict"] = label_SZtype_dict
    info["RecordingDuration"] = duration
    df_all.attrs["info"] = info
    df_all.attrs["data_len"] = data_len
    df_all.attrs["data_shape"] = (fs, n_channels)
    df_all.attrs["segments"] = segments

    # Save the DataFrame and metadata to Parquet
    parquet_subject_dir = os.path.join(parquet_dataset_dir, subject)
    os.makedirs(parquet_subject_dir, exist_ok=True)
    parquet_file = os.path.join(parquet_subject_dir, f"{subject}.parquet")
    df_all.to_parquet(
        parquet_file, index=False, compression="lz4", row_group_size=512
    )

    meta_data_file = os.path.join(parquet_subject_dir, f"{subject}_meta.json")
    with open(meta_data_file, "w") as f:
        json.dump(df_all.attrs, f)


def main():
    """
    Main function to process the dataset and convert it to Parquet format.
    """
    parser = argparse.ArgumentParser(
        description="Convert BIDS dataset to Parquet format."
    )
    parser.add_argument(
        "dataset_dir", type=str, help="Path to the BIDS dataset directory."
    )
    parser.add_argument(
        "parq_dir", type=str, help="Path to the output Parquet directory."
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    parq_dir = args.parq_dir

    layout = BIDSLayout(dataset_dir)
    events_dir = os.path.join(dataset_dir, "events.json")

    # Load seizure types from events.json
    with open(events_dir, "r") as f:
        events = json.load(f)
        sz_types = events["Levels"]

    label_SZtype_dict = {
        index: key for index, (key, value) in enumerate(sz_types.items())
    }

    # Create output directories
    os.makedirs(parq_dir, exist_ok=True)
    parquet_dataset_dir = os.path.join(parq_dir, "CHBMIT")
    os.makedirs(parquet_dataset_dir, exist_ok=True)

    # Process all subjects
    subjects = [f"{i:02}" for i in range(1, 25)]
    for subject in subjects:
        process_subject(
            subject, layout, label_SZtype_dict, sz_types, parquet_dataset_dir
        )


if __name__ == "__main__":
    main()
