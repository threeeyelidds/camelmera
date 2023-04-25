import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import os
import numpy as np
import open3d as o3d

class MathExpressionDataset(Dataset):
    def __init__(self, tokenizer, data, max_length):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        value, expression = self.data[idx]
        input_text = f"{expression}"
        masked_input, masked_labels, attention_mask = create_masked_input_and_labels(input_text, self.tokenizer)

        return {
            "input_ids": masked_input.flatten(),
            "attention_mask": attention_mask.flatten(),
            "masked_labels": masked_labels.flatten(),
            "value_labels": torch.tensor(float(value), dtype=torch.float)
        }

def process_image(image_path):
    image = Image.open(image_path)
    transform_image = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform_image(image)
    return image

def process_depth(depth_path):
    depth = Image.open(depth_path)
    transform_depth = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5/2])
    ])
    depth = transform_depth(depth)
    return depth[1:]

def process_lidar(filename):
  # Load point cloud data from file
  pcd = o3d.io.read_point_cloud(filename)
  points = np.asarray(pcd.points)

  # Set voxel size
  voxel_size = 0.1

  # Voxelization
  voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size=voxel_size)
  voxels = np.asarray(voxel_grid.get_voxels())
  # print(voxels.shape)

  # Extract voxel features
  features = []
  for voxel in voxels:
      voxel_indices = voxel.grid_index
      if len(voxel_indices) == 0:
          feature = np.zeros(6, dtype=np.float32)
      else:
          voxel_points = points[voxel_indices]
          feature = np.concatenate([np.mean(voxel_points[:, :3], axis=0), np.max(voxel_points[:, :3], axis=0)])
      features.append(feature)
  features = np.stack(features)

  # Normalize features
  features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

  # Convert features to tensor
  tensor = torch.from_numpy(features)
  tensor = tensor.permute(1, 0).reshape(-1)  # (batch_size=1, num_channels=6, height=num_voxels, width=1)
  padding=3*224*224-tensor.shape[-1]
  tensor = torch.nn.functional.pad(tensor, (0, padding), mode='constant', value=0).reshape((3,224,224)).float()
  # print(tensor)
#   print(tensor.shape)
  return tensor

class MultimodalDataset(Dataset):
    def __init__(self, data_dir):
        # create a list of image/depth/lidar paths
        self.image_paths = []
        self.depth_paths = []
        self.lidar_paths = []
        # get folders
        for folder in os.listdir(data_dir):
            trajectory_folder_path = os.path.join(data_dir, folder)
            if not os.path.isdir(trajectory_folder_path):
                continue
            print(f'Processing folder: {trajectory_folder_path}')

            self.image_folder_name = 'image_lcam_fish'
            self.depth_folder_name = 'depth_lcam_fish'
            self.lidar_folder_name = 'lidar'
            
            image_folder_path = os.path.join(trajectory_folder_path, self.image_folder_name)
            depth_folder_path = os.path.join(trajectory_folder_path, self.depth_folder_name)
            lidar_folder_path = os.path.join(trajectory_folder_path, self.lidar_folder_name)

            if not os.path.exists(image_folder_path):
                continue
            if not os.path.exists(depth_folder_path):
                continue
            if not os.path.exists(lidar_folder_path):
                continue

            # get image/depth/lidar paths
            if len(os.listdir(image_folder_path)) != len(os.listdir(depth_folder_path)) \
                or len(os.listdir(image_folder_path)) != len(os.listdir(lidar_folder_path)) \
                or len(os.listdir(depth_folder_path)) != len(os.listdir(lidar_folder_path)):
                print(f'Number of images, depth, and lidar files do not match in folder: {trajectory_folder_path}')
                continue
            self.image_paths += [os.path.join(image_folder_path, path) for path in os.listdir(image_folder_path)]
            self.depth_paths += [os.path.join(depth_folder_path, path) for path in os.listdir(depth_folder_path)]
            self.lidar_paths += [os.path.join(lidar_folder_path, path) for path in os.listdir(lidar_folder_path)]
        print(f'Number of images: {len(self.image_paths)}')
        print(f'Number of depth: {len(self.depth_paths)}')
        print(f'Number of lidar: {len(self.lidar_paths)}')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # read the image from disk
        image_path = self.image_paths[index]
        image = process_image(image_path)

        # read the depth from disk
        depth_path = self.depth_paths[index]
        depth = process_depth(depth_path)

        # read the lidar from disk
        lidar_path = self.lidar_paths[index]
        lidar = process_lidar(lidar_path)

        return {
            "pixel_values": image,
            "pixel_values1": depth,
            "pixel_values2": lidar
        }

class MultimodalDatasetPerTrajectory(Dataset):
    def __init__(self, trajectory_folder_path):

        # create a list of image/depth/lidar paths
        self.image_paths = []
        self.depth_paths = []
        self.lidar_paths = []
        self.pose = []
        # get folders
        # for folder in os.listdir(data_dir):
        #     trajectory_folder_path = os.path.join(data_dir, folder)
        # if not os.path.isdir(trajectory_folder_path):
            # continue
        print(f'Processing folder: {trajectory_folder_path}')

        self.image_folder_name = 'image_lcam_fish'
        self.depth_folder_name = 'depth_lcam_fish'
        self.lidar_folder_name = 'lidar'
        
        image_folder_path = os.path.join(trajectory_folder_path, self.image_folder_name)
        depth_folder_path = os.path.join(trajectory_folder_path, self.depth_folder_name)
        lidar_folder_path = os.path.join(trajectory_folder_path, self.lidar_folder_name)
        self.pose_file_path = os.path.join(trajectory_folder_path, 'pose_lcam_front.txt')

        # if not os.path.exists(image_folder_path):
        #     continue
        # if not os.path.exists(depth_folder_path):
        #     continue
        # if not os.path.exists(lidar_folder_path):
        #     continue

        # get image/depth/lidar paths
        with open(self.pose_file_path) as f:
            lines = f.readlines()
            for line in lines:
                pose_list = line.split(" ")
                pose_list = [float(_) for _ in pose_list]
                self.pose.append(torch.Tensor(pose_list))
            # self.pose = f.readlines()
        
        if len(os.listdir(image_folder_path)) != len(os.listdir(depth_folder_path)) \
            or len(os.listdir(image_folder_path)) != len(os.listdir(lidar_folder_path)) \
            or len(os.listdir(depth_folder_path)) != len(os.listdir(lidar_folder_path)) \
            or len(self.pose) != len(os.listdir(image_folder_path)):
            print(f'Number of images, depth, lidar, pose files do not match in folder: {trajectory_folder_path}')
            # continue
        self.image_paths += [os.path.join(image_folder_path, path) for path in os.listdir(image_folder_path)]
        self.depth_paths += [os.path.join(depth_folder_path, path) for path in os.listdir(depth_folder_path)]
        self.lidar_paths += [os.path.join(lidar_folder_path, path) for path in os.listdir(lidar_folder_path)]
        print(f'Number of images: {len(self.image_paths)}')
        print(f'Number of depth: {len(self.depth_paths)}')
        print(f'Number of lidar: {len(self.lidar_paths)}')
        print(f'Number of pose: {len(self.pose)}')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # read the image from disk
        image_path = self.image_paths[index]
        image = process_image(image_path)

        # read the depth from disk
        depth_path = self.depth_paths[index]
        depth = process_depth(depth_path)

        # read the lidar from disk
        lidar_path = self.lidar_paths[index]
        lidar = process_lidar(lidar_path)

        return {
            "pixel_values": image,
            "pixel_values1": depth,
            "pixel_values2": lidar,
            "pose_values": self.pose[index]
        }


