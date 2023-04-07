import boto3
import requests
from urllib.parse import urlparse
from azure.storage.blob import BlobServiceClient
import os

# Create an S3 client
s3_client = boto3.client('s3')

# Specify the name of the S3 bucket and the file to be uploaded
bucket_name = 'yadixawsbucket'


def ss(number:str):

    # Define your Azure Blob Storage connection string and container name
    connection_string = 'DefaultEndpointsProtocol=https;AccountName=tartanairv2;AccountKey=PH1q1TB4fHqGFnvfivj9jPrvABz2ESX1OKkrA4+8G3qoHJpIPDmDok4d2uzDNF7RVR9A4cE7Y00n5nsCrc+edA==;EndpointSuffix=core.windows.net'
    container_name = 'data-raw'

    # Define your local destination folder
    destination_folder_base = f'tartanairv2filtered/AbandonedCableExposure_EASY_P00{number}/'

    # Create a BlobServiceClient object
    blob_service_client = BlobServiceClient.from_connection_string(
        connection_string)

    # Get a reference to the P000 directory
    blob_container_client = blob_service_client.get_container_client(
        container_name)

    directory_name_combinations = ['depth_lcam_fish', 'depth_rcam_fish', 'image_lcam_fish',
                                   'image_rcam_fish', 'imu', 'lidar', 'pose_rcam_front.txt', 'pose_lcam_front.txt']
    # directory_name_combinations = ['dep'lidar']
    name_starts_with_temp = f'AbandonedCableExposure/Data_easy/P00{number}/'

    final_p0000_blob_list_with_sub_directories = []
    for combination in directory_name_combinations:
        final_p0000_blob_list_with_sub_directories.append(
            name_starts_with_temp+combination)

    print(final_p0000_blob_list_with_sub_directories)

    # p000_blob_list_all = []
    for name_starts_with_ in final_p0000_blob_list_with_sub_directories:
        p000_blob_list = blob_container_client.list_blobs(
            name_starts_with=name_starts_with_)
        # p000_blob_list_all.append(p000_blob_list)

        # num = 0
        for blob in p000_blob_list:
            # num += 1
            # if (num > 5):
            #     break
            # for blob in p000_blob_list:
            blob_client = blob_container_client.get_blob_client(blob.name)
            blob_data = blob_client.download_blob()

            # Get the filename from the blob URL
            print(blob_client.url)
            filename = os.path.basename(blob_client.url)
            print(
                f"name_starts_with_ + filename: {name_starts_with_} + {filename}")

            # Move the file to your local machine
            if (name_starts_with_.endswith('txt')):
                destination_folder = destination_folder_base
            else:
                destination_folder = destination_folder_base + \
                    name_starts_with_.split("/")[-1]
            destination_file = os.path.join(destination_folder, filename)

            # Create the parent directory (and any intermediate directories) if it doesn't exist
            os.makedirs(destination_folder, exist_ok=True)

            print(f"destination_file: {destination_file}")
            with open(destination_file, "wb") as file:
                file.write(blob_data.readall())

            # Upload the file to the S3 bucket
            s3_client.upload_file(destination_file, bucket_name, destination_file)


ss('2')
