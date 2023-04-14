# Azure Blob Storage Downloader

This Python script downloads files from an Azure Blob Storage container and saves them locally. The script requires an Azure Blob Storage connection string and a container name to access the files. The downloaded files are filtered by a list of desired paths and saved in a specified destination folder.

## Prerequisites
Before running the script, you must have the following:

+ Python 3.x installed on your machine.
+ The following Python packages installed: urllib, azure-storage-blob.

## Setup

1. Clone or download the repository.
2. Install the required Python packages using the following command, `requirements.txt` is under the `/data` folder:
```
pip install -r requirements.txt
```

## Running the Script
```
python download.py
```
The script will start downloading files that match the desired paths and save them in the specified destination folder. The script prints the name of each file it downloads and the full path where it saves the file.