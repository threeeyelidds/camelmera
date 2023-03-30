import gym
import numpy as np
import torch
import wandb

import argparse
import pickle
import random
import sys
import pickle

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.training.seq_trainer import SequenceTrainer


import numpy as np
import random

import os
import numpy as np
from PIL import Image
import torch
import timm
from torchvision import transforms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_fish_depth(main_folder_path, goal_position):
    all_datasets = []

    for folder in os.listdir(main_folder_path):
        folder_path = os.path.join(main_folder_path, folder)

        if not os.path.isdir(folder_path):
            continue

        states = []
        actions = []
        rewards = []
        positions = []
        
        print(f'Processing folder: {folder_path}') 

        pose_file_path = os.path.join(folder_path, 'pose_lcam_front.txt')
        if not os.path.exists(pose_file_path):
            continue

        with open(pose_file_path) as f:
            for line in f.readlines():
                values = line.strip().split()
                x, y, z = map(float, values[:3])
                positions.append(np.array([x, y, z]))

        model_name = 'vit_base_patch16_224'
        model = timm.create_model(model_name, pretrained=True)
        model.eval()

        preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        image_folder_path = os.path.join(folder_path, 'image_lcam_fish')
        if not os.path.exists(image_folder_path):
            continue
        depth_folder_path = os.path.join(folder_path, 'depth_lcam_fish')
        if not os.path.exists(depth_folder_path):
            continue

        positions = np.array(positions)  # Convert positions to a numpy array

        images = sorted(os.listdir(image_folder_path))
        depths = sorted(os.listdir(depth_folder_path))
        for idx in range(len(images)):
            if images[idx].endswith('.png'):
                img_path = os.path.join(image_folder_path, images[idx])
                img = Image.open(img_path)
                depth_path = os.path.join(depth_folder_path, depths[idx])
                depth = Image.open(depth_path)

                input_tensor = preprocess(img).concatenate(preprocess(depth), dim=0)
                input_batch = input_tensor.unsqueeze(0)
                with torch.no_grad():
                    embedding = model(input_batch)

                embedding = embedding.squeeze().numpy()

                state = np.hstack((embedding, positions[idx]))  # Stack the embeddings and positions horizontally
                states.append(state)

                if idx > 0:
                    action = positions[idx] - positions[idx - 1]
                    actions.append(action)

                    reward = -np.linalg.norm(positions[idx] - goal_position) # Labeling Reward as negative distance to goal
                    rewards.append(reward)

        dataset = {
            'observations': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards)
        }       


        all_datasets.append(dataset)

    return all_datasets