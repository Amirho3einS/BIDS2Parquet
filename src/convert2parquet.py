import os

import pandas as pd
import pyedflib

from epilepsy2bids.annotations import Annotations

tsv_dir = ""

annot = Annotations.loadTsv(tsv_dir)
labels = annot.getMask(fs=1)

# Load EDF file
edf_file = ""
with pyedflib.EdfReader(edf_file) as f:
    header = f.getHeader()
    n_channels = header["n_channels"]
    data = [f.readSignal(i) for i in range(n_channels)]
    data = pd.DataFrame(data).T

# TODO Meta data
# Channel names, sampling rate, other headers...


def saveDataFrame(file: str):
    """Save Eeg object to a dataframe compatible file.

    Args:
        file (str):  path of the file to save to. If directory does not exist it is created.
        format (FileFormat, optional): File format to save to. Defaults to FileFormat.PARQUET_GZIP.

    Raises:
        ValueError: raised if fileFormat is not supported.
    """
    tmpData = self.data.copy()
    dataDF = pd.DataFrame(data=tmpData.transpose(), columns=self.channels)
    # TODO save metadata -- pyarrow might be a good candidate
    dataDF.attrs["fileHeader"] = self._fileHeader
    dataDF.attrs["fileHeader"]["startdate"] = str(
        dataDF.attrs["fileHeader"]["startdate"]
    )
    dataDF.attrs["signalHeader"] = self._signalHeader

    # Create directory for file
    if os.path.dirname(file):
        os.makedirs(os.path.dirname(file), exist_ok=True)
    # Write new file
    dataDF.to_parquet(
        file,
        index=False,
        compression="lz4",
    )
    return file
