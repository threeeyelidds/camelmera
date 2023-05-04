import airsim
import numpy as np
import gym
from gym import spaces
from PIL import Image
import torch
import timm
from torchvision import transforms
from scipy.spatial.transform import Rotation as R
import open3d as o3d

from custom_models.CustomViT import CustomViT
from custom_models.CustomViTMAE import CustomViTMAE

from transformers import AutoImageProcessor, ViTMAEForPreTraining, ViTMAEConfig

import cv2
from tartanair.sample_pipeline.src.postprocessing.SimulatedLiDAR import SimulatedLiDAR
from tartanair.sample_pipeline.src.postprocessing.SimulatedLiDARModel import VELODYNE_VLP_16
from tartanair.sample_pipeline.src.postprocessing.SimulatedLiDAR import convert_DEA_2_Velodyne_XYZ


# Functions to get sythetic lidar data

def read_compressed_float(image):
    return np.squeeze(image.view('<f4'), axis=-1)


def get_depth_change(depth_image, threshold=0.1):
    depthnp = depth_image.astype(np.float32)

    gx = cv2.Sobel(depthnp, cv2.CV_32FC1, 2, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
    gy = cv2.Sobel(depthnp, cv2.CV_32FC1, 0, 2, ksize=3, borderType=cv2.BORDER_REFLECT)

    # Sum (norm) the results.
    s = np.sqrt(gx**2 + gy**2)

    # adaptive threshold wrt the depth value
    adaptive_thresh = np.clip(depthnp.astype(np.float32) * threshold, threshold, 10)

    # Find the over-threshold ones.
    m = s > adaptive_thresh  # self.threshold
    m = m.astype(np.float32)

    return m


def get_lidar_points_from_depth(depth_image):
    try:
        print("=====depth_image shape:", depth_image.shape, "=====depth_image dtype:", depth_image.dtype)

        depth_image = read_compressed_float(depth_image)

        depth_change = get_depth_change(depth_image)
        print("=====depth_change shape:", depth_change.shape)

        sld = SimulatedLiDAR(320, 640)
        sld.set_description(VELODYNE_VLP_16)
        sld.initialize()

        lidar_points = sld.extract([depth_image], depthChangeMask=[depth_change])
        lidar_points = lidar_points.reshape((-1, 3))
        xyz = convert_DEA_2_Velodyne_XYZ(lidar_points[:, 0], lidar_points[:, 1], lidar_points[:, 2])
        print("=====xyz shape:", xyz.shape)

        return xyz
    except Exception as e:
        print("Error inside get_lidar_points_from_depth:", e)
        raise

# Helper function

def process_lidar(points):
    # pcd = o3d.io.read_point_cloud('/content/drive/MyDrive/000000_lcam_front_lidar.ply')
    # points = np.asarray(pcd.points)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    # Set the voxel size
    voxel_size = 0.6

    # Downsample the point cloud
    downsampled_point_cloud = point_cloud.voxel_down_sample(voxel_size)
    # Create an Open3D VoxelGrid object
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(downsampled_point_cloud, voxel_size)
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
    #  print(tensor.shape)
    return tensor

def process_image(image):
    transform_image = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform_image(image)
    return image

def process_depth(depth):
    transform_depth = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5/2])
    ])
    depth = transform_depth(depth)
    return depth[1:]

"""
The step function takes an action index as input and calls the _do_action function to execute the action in the AirSim environment.
The _do_action function calls interpret_action to get the position offsets and orientation changes corresponding to the input action index.
The current position and orientation of the drone are retrieved from the AirSim environment.
The position and orientation are updated based on the action offsets and changes.
The drone is moved to the new position and orientation in the AirSim environment.
"""
class AirSimDroneEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, ip_address, ob_min, ob_max, goal, step_length=10, threshold=0.05, goal_reward=100, max_steps=500, start_position=[-0.5, 0, -1]):
        '''
        goal: the goal position of the drone
        '''
        self.observation_space = spaces.Box(low=0, high=1, shape=(768,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1, -1, -1, -0.1, -0.1, -0.1, -0.1]),
                                    high=np.array([1, 1, 1, 0.1, 0.1, 0.1, 0.1]),
                                    dtype=np.float32)
        self.viewer = None
        self.step_length = step_length
        self.ob_min = ob_min
        self.ob_max = ob_max
        self.goal = goal
        self.threshold = threshold
        self.goal_reward = goal_reward
        self.max_steps = max_steps
        self.steps = 0
        self.position = start_position
        self.start_position = start_position

        output_dir='C:/Users/Tianyi/Desktop/11777/camelmera/models/gym/multimodal'
        # trained_model_name = 'multimodal'
        # output_dir='/home/ubuntu/weights/' + trained_model_name

        # Initialize a new CustomViT model
        model_name = "facebook/vit-mae-base"
        vit_config = ViTMAEConfig.from_pretrained(model_name)
        vit_config.output_hidden_states=True
        vit_model = CustomViT(config=vit_config)

        # Initialize a new CustomViTMAE model
        model_name = "facebook/vit-mae-base"
        config = ViTMAEConfig.from_pretrained(model_name)
        config.output_hidden_states=True
        custom_model = CustomViTMAE(config=config)
        custom_model.vit = vit_model

        # Load the state_dict from the saved model
        state_dict = torch.load(f"{output_dir}/pytorch_model.bin", map_location=torch.device('cpu'))

        custom_model.load_state_dict(state_dict)

        # don't need decoders
        self.vit_encoder = custom_model.vit

        # initialize image_request and depth_request
        self.image_request = airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        self.depth_request = airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True)

        self.drone = airsim.MultirotorClient(ip=ip_address, timeout_value=3600)
        self.drone.confirmConnection()
        print("Vehicles: ", self.drone.listVehicles())
        self._setup_flight()

        # initialize the state
        self.state = self._get_obs()



    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        # Set home position and velocity
        x,y,z = self.start_position
        self.drone.moveToPositionAsync(x,y,z, 10).join()
        self.drone.moveByVelocityAsync(1, -0.67, -0.8, 5).join()
    
    def _get_obs(self):
        # Get the images
        responses = self.drone.simGetImages([
        airsim.ImageRequest(0, airsim.ImageType.Scene, False, False),
        airsim.ImageRequest(1, airsim.ImageType.DepthPlanar, True, False)
        ])

        # Extract and preprocess image, depth, and lidar data
        # print("========Image dimensions========: ", responses[0].height, responses[0].width)
        image_data = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        image_data = image_data.reshape(responses[0].height, responses[0].width, 3)[:, :, :3]

        # print("========Depth dimensions========: ", responses[1].height, responses[1].width)
        depth_data = np.frombuffer(bytearray(np.array(responses[1].image_data_float, dtype=np.float32)), dtype=np.float32)
        depth_data = np.reshape(depth_data, (responses[1].height, responses[1].width, 1))
        # print("========Depth dimensions reshaped========: ", responses[1].height, responses[1].width)

        try:
            lidar_points = get_lidar_points_from_depth(depth_data)
        except Exception as e:
            print("========Error obtaining LiDAR data from depth image:", e)


        # Process the LiDAR points
        if len(lidar_points) > 0:
            # print("==============Lidar tensor is not empty=========")
            lidar_tensor = process_lidar(lidar_points).unsqueeze(0)
            print("============lidar shape=========",lidar_tensor.shape)
        else:
            # print("===============lidar tensor is empty==========")
            lidar_tensor = torch.zeros(3, 224, 224, dtype=torch.float32)  # Default empty tensor


        # Convert image_data and depth_data to PIL images
        image_pil = Image.fromarray(image_data)
        depth_pil = Image.fromarray(depth_data.squeeze(), mode='F')

        # Preprocess image and depth data
        image_tensor = process_image(image_pil).unsqueeze(0)
        print("========Image dimensions reshaped========: ", image_tensor.shape)

        depth_tensor = process_depth(depth_pil).unsqueeze(0)
        print("========Depth dimensions reshaped========: ", depth_tensor.shape)

        depth_tensor = torch.zeros(1, 3, 224, 224)
       
        # Extract features from the input_data using the ViT model
        self.vit_encoder.eval()
        with torch.no_grad():
            outputs = self.vit_encoder(image_tensor,depth_tensor,lidar_tensor)
            embedding = outputs.last_hidden_state[:, 0, :]
            print(embedding.shape)

        observations = embedding.detach().numpy()
        print("embedding_size out of Custom ViT", embedding.shape)
        normalized_observations = (observations - self.ob_min) / (self.ob_max - self.ob_min)
        return normalized_observations

    def _compute_reward(self):
        # Define the goal state embedding
        distance = np.linalg.norm(self.state - self.goal)

        if distance <= self.threshold:
            # Give a large positive reward when the goal is reached
            reward = self.goal_reward
            done = False
        else:
            # Give a negative reward proportional to the distance otherwise
            reward = -distance
            done = True

        return reward, done

    def _do_action(self, action):
        position_difference, quaternion_changes = action[0,:3], action[0,3:]
        
        # Get the current position and orientation of the drone
        drone_pose = self.drone.simGetVehiclePose()
        current_position = np.array(drone_pose.position.to_numpy_array())
        current_quaternion = np.array([drone_pose.orientation.w_val, drone_pose.orientation.x_val, drone_pose.orientation.y_val, drone_pose.orientation.z_val])

        # Update the position and quaternion based on the action
        print(position_difference)
        print(current_position)
        new_position = current_position + position_difference
        new_quaternion = self._apply_quaternion_changes(current_quaternion, quaternion_changes)

        # Move the drone to the new position and orientation
        # self.drone.enableApiControl(True)
        # self.drone.armDisarm(True)
        x,y,z=new_position
        x,y,z=float(x),float(y),float(z)
        self.drone.moveToPositionAsync(x,y,z, 10).join()
        a,b,c,d = new_quaternion
        a,b,c,d = float(a),float(b),float(c),float(d)
        self.drone.simSetVehiclePose(
            airsim.Pose(
                airsim.Vector3r(x,y,z),
                airsim.Quaternionr(b,c,d,a),
            ),
            True,
        )
        # update position
        self.position = new_position

    def step(self, action):
        """
        obs: The next state or observation after taking the specified action. In this case, it is a dictionary containing the "embedding" and "position" of the drone.
        reward: A scalar value representing the immediate reward obtained after taking the action. In this case, it is the negative L2 norm between the current state embedding and the goal state embedding.
        done: A boolean value indicating whether the episode has ended or not. In this case, it is True if there is a collision, and False otherwise.
        info: A dictionary containing additional information about the environment. In this case, it is the current state dictionary containing "position", "collision", and "prev_position".
        """
        self._do_action(action)
        self.state = self._get_obs()
        reward, done = self._compute_reward()
        self.steps += 1
        if self.steps >= self.max_steps:
            done = True
        info = ""
        return self.state, reward, done, info

    def reset(self):
        self._setup_flight()
        self.steps = 0
        self.state = self._get_obs()
        self.position = self.start_position
        return self.state

    def render(self):
        pass

    def close(self):
        self.drone.reset()

    def _apply_quaternion_changes(self, current_quaternion, quaternion_changes):
        r1 = R.from_quat(current_quaternion)
        r2 = R.from_quat(quaternion_changes)
        new_rotation = r1 * r2
        return new_rotation.as_quat()
    
    def get_position(self):
        return self.position