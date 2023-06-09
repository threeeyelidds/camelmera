import airsim
import numpy as np
import math
import time
from argparse import ArgumentParser

import gym
from gym import spaces
from PIL import Image

import torch
import timm
from torchvision import transforms

class AirSimEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, image_shape):
        self.observation_space = spaces.Box(0, 255, shape=image_shape, dtype=np.uint8)
        self.viewer = None

    def __del__(self):
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _compute_reward(self):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def step(self, action):
        raise NotImplementedError()

    def render(self):
        return self._get_obs()

class AirSimDroneEnv(AirSimEnv):
    def __init__(self, ip_address, step_length, image_shape):
        super().__init__(image_shape)
        self.step_length = step_length
        self.image_shape = image_shape

        self.state = {
            "position": np.zeros(3),
            "collision": False,
            "prev_position": np.zeros(3),
        }

        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.action_space = spaces.Discrete(7)
        self._setup_flight()

        self.image_request = airsim.ImageRequest(
            3, airsim.ImageType.DepthPerspective, True, False
        )

        # Initialize the pretrained Vision Transformer
        model_name = 'vit_base_patch16_224'
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.eval()

    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        # Set home position and velocity
        self.drone.moveToPositionAsync(-0.55265, -31.9786, -19.0225, 10).join()
        self.drone.moveByVelocityAsync(1, -0.67, -0.8, 5).join()


        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert("L"))

        return im_final.reshape([84, 84, 1])
    def _get_obs(self):
        # Get the images
        responses = self.drone.simGetImages([self.image_request, self.left_camera_request])

        # Converting the raw image data received from the left camera into a NumPy array and then extracting the RGB channels from it
        left_camera_image = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8)
        left_camera_image = left_camera_image.reshape(responses[1].height, responses[1].width, 4)[:, :, :3]

        # Preprocess the image
        preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        img = Image.fromarray(left_camera_image)
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0)



        # Extract features from the image using the ViT model
        with torch.no_grad():
            embedding = self.model(input_batch)

        embedding = embedding.squeeze().numpy()

        # Concatenate the embeddings and the agent's position
        position = np.array([
            self.state["position"].x_val,
            self.state["position"].y_val,
            self.state["position"].z_val
        ])
        observation = {
            "embedding": np.hstack((embedding, position)),
            "position": position
        }

        return observation


    def _do_action(self, action):
        quad_offset = self.interpret_action(action)
        quad_vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        self.drone.moveByVelocityAsync(
            quad_vel.x_val + quad_offset[0],
            quad_vel.y_val + quad_offset[1],
            quad_vel.z_val + quad_offset[2],
            5,
        ).join()

    def _compute_reward(self):
        # Define the goal position
        goal_position = np.array([5, 5, 5])

        # Calculate the negative Euclidean distance
        quad_pt = np.array([
            self.state["position"].x_val,
            self.state["position"].y_val,
            self.state["position"].z_val
        ])
        reward = -np.linalg.norm(quad_pt - goal_position)

        # Check for collision
        done = self.state["collision"]

        return reward, done


    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()

        return obs, reward, done, self.state

    def reset(self):
        self._setup_flight()
        return self._get_obs()

    def interpret_action(self, action):
        if action == 0:
            quad_offset = (self.step_length, 0, 0)
        elif action == 1:
            quad_offset = (0, self.step_length, 0)
        elif action == 2:
            quad_offset = (0, 0, self.step_length)
        elif action == 3:
            quad_offset = (-self.step_length, 0, 0)
        elif action == 4:
            quad_offset = (0, -self.step_length, 0)
        elif action == 5:
            quad_offset = (0, 0, -self.step_length)
        else:
            quad_offset = (0, 0, 0)

        return quad_offset
