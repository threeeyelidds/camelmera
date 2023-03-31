import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pickle


class IMU_Image_Dataset(Dataset):
    def __init__(self, goal_position, load_from_file=False, preprocessed_data_path=None, data_dir=None, image_folder_name=None, transform=None):
        self.datasets = []
        self.image_folder_name = image_folder_name
        if load_from_file:
            if preprocessed_data_path != None and os.path.exists(preprocessed_data_path):
                print('Loading preprocessed data from file...')
                with open(preprocessed_data_path, 'rb') as f:
                    self.datasets = pickle.load(f)
            else:
                load_from_file = False
        if not load_from_file:
            if os.path.exists(data_dir):
                print('Preprocessing data...')
                self.datasets = self.load_data(data_dir, goal_position)
                print('Saving preprocessed data to file...')
                if preprocessed_data_path == None:
                    preprocessed_data_path = "./data/preprocessed_data_v0.pkl"
                self.save_preprocessed_data(self.datasets, preprocessed_data_path)
            else:
                print('Can not get preprocessed data...')

    def get_all_imu_path(self, imu_dir):
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

    def get_imu_data_at_one_time(self, i:int, imu_all_data):
        result_list = []
        for imu in imu_all_data:
            result_list.append(imu[i])
        return np.concatenate(result_list, axis=0)
    
    def get_imu_data(self, cam_time, imu_time, imu_all_data):
        imu_data_list = []
        max_lenth = 0
        j = 0
        for i in range(cam_time.shape[0]-1)[:]:
            now_img_time = cam_time[i]
            next_img_time = cam_time[i+1]
            arrays_to_concatenate = []
            # print(i)
            while ( (j < imu_time.shape[0]) and (now_img_time <= imu_time[j] < next_img_time) ):
                imu_data_at_one_time = self.get_imu_data_at_one_time(i, imu_all_data)
                # print(imu_data_at_one_time.shape)
                arrays_to_concatenate.append(imu_data_at_one_time) # TODO: (24,)
                j += 1
            result = np.stack(arrays_to_concatenate, axis=1) # axis TBD # this is the imu data for time i
            # print(result.shape) # (24, 9~11)
            if result.shape[1] > max_lenth:
                max_lenth = result.shape[1]
            imu_data_list.append(result)

        print("imu_data_list len:", len(imu_data_list))

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
        print("Shape of the resulting array:", result_array.shape)

        imu_data_list = result_array  # Convert imu_data_list to a numpy array
        print("imu_data_list.shape:", imu_data_list.shape)
        return imu_data_list

    def load_data(self, data_dir, goal_position):
        all_datasets = []

        for folder in os.listdir(data_dir):
            trajectory_folder_path = os.path.join(data_dir, folder)
            if not os.path.isdir(trajectory_folder_path):
                continue
            states = []
            actions = []
            rewards = []
            positions = []
            print(f'Processing folder: {trajectory_folder_path}')

            imu_dir = os.path.join(trajectory_folder_path, "imu")
            all_imu_path = self.get_all_imu_path(imu_dir)
            cam_time_path = os.path.join(imu_dir, "cam_time.npy")
            imu_time_path = os.path.join(imu_dir, "imu_time.npy")
            if not os.path.exists(cam_time_path):
                print("cam time file not found:", cam_time_path)
                continue
            if not os.path.exists(imu_time_path):
                print("imu time file not found:", imu_time_path)
                continue

            if self.image_folder_name == None:
                self.image_folder_name = 'image_lcam_fish'
            image_folder_path = os.path.join(trajectory_folder_path, self.image_folder_name)
            pose_file_path = os.path.join(trajectory_folder_path, 'pose_lcam_front.txt')

            imu_all_data = []
            imu_data_len = -1

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

            print("cam_time.shape:", cam_time.shape)
            print("imu_time.shape:", imu_time.shape)

            imu_data_list = self.get_imu_data(cam_time, imu_time, imu_all_data)

            with open(pose_file_path) as f:
                for line in f.readlines():
                    values = line.strip().split()
                    x, y, z = map(float, values[:3])
                    positions.append(np.array([x, y, z]))
            positions = np.array(positions)  # Convert positions to a numpy array

            preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
            model_name = 'vit_base_patch16_224'
            vit_model = timm.create_model(model_name, pretrained=True)
            vit_model.eval()


            for idx, img_name in enumerate(sorted(os.listdir(image_folder_path))):
                print(idx, img_name)
                if img_name.endswith('.png'):
                    img_path = os.path.join(image_folder_path, img_name)
                    img = Image.open(img_path)

                    input_tensor = preprocess(img)
                    # input_tensor = preprocess(img).concatenate(preprocess(depth), dim=0)
                    input_batch = input_tensor.unsqueeze(0)
                    with torch.no_grad():
                        embedding = vit_model.forward_features(input_batch)
                        # embedding = vit_model(input_batch)
                        print(embedding.shape)
                    # embedding = torch.cat(embedding, dim=-1)
                    embedding = embedding.reshape(-1).numpy()
                    # embedding = embedding.squeeze().numpy()

                    print(embedding.shape, positions[idx].shape, imu_data_list[idx].reshape(-1).shape)

                    state = np.hstack((embedding, positions[idx], imu_data_list[idx].reshape(-1)))  # Stack the embeddings and positions horizontally
                    states.append(state)

                    if idx > 0:
                        action = positions[idx] - positions[idx - 1]
                        actions.append(action)

                        reward = -np.linalg.norm(positions[idx] - goal_position) # Labeling Reward as negative distance to goal
                        rewards.append(reward)
            all_datasets.append({
                      'observations': np.array(states),
                      'actions': np.array(actions),
                      'rewards': np.array(rewards)
                      })
            print("all_datasets len", len(all_datasets))
        return all_datasets


    def save_preprocessed_data(self, datasets, preprocessed_data_path):
        with open(preprocessed_data_path, 'wb') as f:
          pickle.dump(datasets, f)
          print(f'Saved preprocessed data {len(datasets)} to {preprocessed_data_path}')

    def __len__(self):
        # Return the total number of images
        return len(self.datasets)

    def __getitem__(self, idx):
        return self.datasets[idx]

if __name__ == "__main__":
    # Example usage of the custom dataset and dataloader
    # Define the image directory and labels
    data_dir = './data'
    goal_position = np.array([10, 10, 10])  

    # Create an instance of the custom dataset
    dataset = IMU_Image_Dataset(goal_position=goal_position, load_from_file=True, preprocessed_data_path="./data/preprocessed_data_v0.pkl", data_dir=data_dir, transform=None)

    # Create a dataloader for batching and shuffling
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # Iterate over the dataloader during training
    for batch_idx, data in enumerate(dataloader):
        # Perform training operations on the batch of data
        print(batch_idx)
        print(len(data))
        print(data['observations'].shape)
        print(data['actions'].shape)
        print(data['rewards'].shape)
        pass

    # Or other self defined dataloader
    trajectories = IMU_Image_Dataset(goal_position=goal_position, load_from_file=True, preprocessed_data_path="./data/preprocessed_data_v0.pkl", data_dir=data_dir, transform=None)
    print(trajectories[0])
    def get_dimensions(dataset):
        state_example, action_example = dataset['observations'][0], dataset['actions'][0]
        state_dim = state_example.shape[0]
        action_dim = action_example.shape[0]

        return state_dim, action_dim
    state_dim, act_dim = get_dimensions(trajectories[0])
    print(state_dim, act_dim)
    # following is the same as experiment.py

