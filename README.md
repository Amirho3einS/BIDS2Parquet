
## About BIDS2Parquet

BIDS2Parquet is a tool designed to convert datasets formatted in the Brain Imaging Data Structure (BIDS) into Parquet files. This conversion facilitates efficient data storage and processing, making it easier to work with large-scale datasets. The tool is specifically tailored for the CHB-MIT Scalp EEG Database (CHB-MIT), a widely used dataset in seizure detection and EEG analysis research.

## Features

- Converts BIDS-formatted EEG data into Parquet files.
- Optimized for handling the CHB-MIT dataset.
- Simplifies data preprocessing for machine learning and data analysis workflows.
- Supports scalable and efficient data storage.

## Installation

To use BIDS2Parquet, clone the repository and install the required dependencies:

```bash
git clone https://github.com/Amirho3einS/BIDS2Parquet.git
cd BIDS2Parquet
pip install -r requirements.txt
```

## Usage

Run the tool to convert a BIDS dataset into Parquet format:

```bash
python generate_parquet.py --dataset_dir /path/to/bids_dataset --parq_dir /path/to/output_directory
```
