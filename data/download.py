from urllib.parse import urlparse
from azure.storage.blob import BlobServiceClient
import os


# default setting would download all trajacories to /data/tartanairv2filtered/
destination_folder_base = 'tartanairv2filtered'

# Define your Azure Blob Storage connection string and container name
connection_string = 'DefaultEndpointsProtocol=https;AccountName=tartanairv2;AccountKey=PH1q1TB4fHqGFnvfivj9jPrvABz2ESX1OKkrA4+8G3qoHJpIPDmDok4d2uzDNF7RVR9A4cE7Y00n5nsCrc+edA==;EndpointSuffix=core.windows.net'
container_name = 'data-raw'


def download():

    # Create a BlobServiceClient object
    blob_service_client = BlobServiceClient.from_connection_string(
        connection_string)

    # Get a reference to the 'data-raw' directory
    blob_container_client = blob_service_client.get_container_client(
        container_name)

    # path_combinations = ['depth_lcam_fish', 'depth_rcam_fish', 'image_lcam_fish',
    #                      'image_rcam_fish', 'imu', 'lidar', 'pose_rcam_front.txt', 'pose_lcam_front.txt']
    path_combinations = ['depth_lcam_fish', 'image_lcam_fish', 'imu', 'lidar', 'pose_lcam_front.txt']

    blob_list = blob_container_client.list_blobs()

    for blob in blob_list:
        path_lists = blob.name.split('/')
        if len(path_lists) > 3 and path_lists[1] == "Data_easy" and path_lists[3] in path_combinations:
            print(blob.name) #e.g., AbandonedCableExposure/Data_easy/P000/depth_lcam_fish/000860_lcam_fish_depth.png

            blob_client = blob_container_client.get_blob_client(blob.name)
            blob_data = blob_client.download_blob()

            filename = blob.name.split("/")[-1]
            trajectory_folder = os.path.join(destination_folder_base, f'{path_lists[0]}_{path_lists[1]}_{path_lists[2]}')

            # Move the file to your local machine
            if (path_lists[3].endswith('txt')):
                destination_folder = trajectory_folder
            else:
                destination_folder = os.path.join(trajectory_folder, path_lists[3])
            destination_file = os.path.join(destination_folder, filename)

            # Create the parent directory (and any intermediate directories) if it doesn't exist
            os.makedirs(destination_folder, exist_ok=True)

            print(f"destination_file: {destination_file}")
            with open(destination_file, "wb") as file:
                file.write(blob_data.readall())


if __name__ == '__main__':
    download()
