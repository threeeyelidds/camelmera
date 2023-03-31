# generate imu.npy at trajectory_folder_path
# imu.npy is a numpy array with the shape of # (image_number, 264)

import os
import numpy as np

data_dir = '/content/drive/MyDrive/tartanairv2filtered'

def get_all_imu_path(imu_dir):
    acc_path = os.path.join(imu_dir, "acc.npy") # accelerometer, three-dimensional space
    acc_nograv_path = os.path.join(imu_dir, "acc_nograv.npy") # accelerometer sensor after removing the contribution of gravity
    acc_nograv_body_path = os.path.join(imu_dir, "acc_nograv_body.npy") # and transforming the measurement into the body frame of reference
    gyro_path = os.path.join(imu_dir, "gyro.npy") # gyroscope sensor. A gyroscope is a device that measures the rate of rotation or angular velocity of an object around a particular axis. measure the rotational motion of an object in three dimensions (yaw, pitch, and roll).
    ori_global_path = os.path.join(imu_dir, "ori_global.npy") # gyroscope sensor. A gyroscope is a device that measures the rate of rotation or angular velocity of an object around a particular axis. measure the rotational motion of an object in three dimensions (yaw, pitch, and roll).
    pos_global_path = os.path.join(imu_dir, "pos_global.npy")
    vel_body_path = os.path.join(imu_dir, "vel_body.npy") # linear velocity of an object in the body frame of reference.
    vel_global_path = os.path.join(imu_dir, "vel_global.npy")
    all_path = [acc_path,
                acc_nograv_path,
                acc_nograv_body_path,
                gyro_path,
                ori_global_path,
                pos_global_path,
                vel_body_path,
                vel_global_path]
    return all_path

def get_imu_data_at_one_time(i:int, imu_all_data):
    result_list = []
    for imu in imu_all_data:
        result_list.append(imu[i])
    return np.concatenate(result_list, axis=0)

def get_imu_data(cam_time, imu_time, imu_all_data):
    imu_data_list = []
    max_lenth = 0
    j = 0
    for i in range(cam_time.shape[0]-1)[:]:
        now_img_time = cam_time[i]
        next_img_time = cam_time[i+1]
        arrays_to_concatenate = []
        # print(i)
        while ( (j < imu_time.shape[0]) and (now_img_time <= imu_time[j] < next_img_time) ):
            imu_data_at_one_time = get_imu_data_at_one_time(i, imu_all_data)
            # print(imu_data_at_one_time.shape)
            arrays_to_concatenate.append(imu_data_at_one_time) # TODO: (24,)
            j += 1
        result = np.stack(arrays_to_concatenate, axis=1) # axis TBD # this is the imu data for time i
        # print(result.shape) # (24, 9~11)
        if result.shape[1] > max_lenth:
            max_lenth = result.shape[1]
        imu_data_list.append(result)

    # print("imu_data_list len:", len(imu_data_list))

    # Determine the number of arrays in the list
    num_arrays = len(imu_data_list)

    # Create an empty array of zeros with the desired shape (num_arrays, 24, 11)
    result_array = np.zeros((num_arrays, 24, max_lenth))

    # Iterate through the list of arrays and copy their content into the result array
    for i, arr in enumerate(imu_data_list):
        # Get the shape of the current array
        rows, cols = arr.shape
        
        # Copy the content of the current array into the corresponding slice of the result array
        result_array[i, :rows, :cols] = arr

    # Print the result and its shape
    # print("Resulting array:\n", result_array)
    # print("Shape of the resulting array:", result_array.shape)

    imu_data_list = result_array  # Convert imu_data_list to a numpy array
    # print("imu_data_list.shape:", imu_data_list.shape)
    return imu_data_list

for folder in os.listdir(data_dir):
    trajectory_folder_path = os.path.join(data_dir, folder)
    if not os.path.isdir(trajectory_folder_path):
        continue
    print(f'\nProcessing folder: {trajectory_folder_path}')

    imu_dir = os.path.join(trajectory_folder_path, "imu")
    all_imu_path = get_all_imu_path(imu_dir)
    cam_time_path = os.path.join(imu_dir, "cam_time.npy")
    imu_time_path = os.path.join(imu_dir, "imu_time.npy")
    if not os.path.exists(cam_time_path):
        print("cam time file not found:", cam_time_path)
        continue
    if not os.path.exists(imu_time_path):
        print("imu time file not found:", imu_time_path)
        continue

    imu_all_data = []
    imu_data_len = -1

    # print("all_imu_path:", all_imu_path)

    for imu_path in all_imu_path:
        if not os.path.exists(imu_path):
            print("imu file not found:", imu_path)
            continue
        imu_data = np.load(imu_path)
        imu_all_data.append(imu_data)
        if imu_data_len == -1:
            imu_data_len = imu_data.shape[0]
        else:
            if imu_data_len != imu_data.shape[0]:
                print("wrong!")

    cam_time = np.load(cam_time_path)
    imu_time = np.load(imu_time_path)

    # print("cam_time.shape:", cam_time.shape)
    # print("imu_time.shape:", imu_time.shape)

    imu_data_list = get_imu_data(cam_time, imu_time, imu_all_data)

    flatten_imu_data = imu_data_list.reshape(imu_data_list.shape[0], -1)

    print(f"saving to {trajectory_folder_path}/imu.npy... flatten_imu_data.shape:", flatten_imu_data.shape)

    np.save(os.path.join(trajectory_folder_path, "imu.npy"), flatten_imu_data) # (image_number, 264)
