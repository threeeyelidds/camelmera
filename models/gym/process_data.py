import pickle
import os
import numpy as np
from PIL import Image
import torch
import timm
from torchvision import transforms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def save_image_pkl(main_folder_path):
    for folder in os.listdir(main_folder_path):
        folder_path = os.path.join(main_folder_path, folder)

        if not os.path.isdir(folder_path):
            continue
        
        print(f'Processing: {folder_path}') 

        preprocess_image = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        image_folder_path = os.path.join(folder_path, 'image_lcam_fish')
        if not os.path.exists(image_folder_path):
            print("Image folder not found")
            continue

        images = sorted(os.listdir(image_folder_path))
        print(len(images))
        img_batch_list = []
        for idx in range(len(images)):
            if images[idx].endswith('.png'):
                img_path = os.path.join(image_folder_path, images[idx])
                # (1000, 1000)
                img = Image.open(img_path)
                img_tensor = preprocess_image(img)
                img_batch = img_tensor.unsqueeze(0)
                print(img_batch.shape)
                img_batch_list.append(img_batch)

        save_file_path = folder_path + '/image_lcam_fish.pkl'
        with open(save_file_path, 'wb') as f:
            pickle.dump(img_batch_list, f)


def save_depth_pkl(main_folder_path):
    for folder in os.listdir(main_folder_path):
        folder_path = os.path.join(main_folder_path, folder)

        if not os.path.isdir(folder_path):
            continue
        
        print(f'Processing: {folder_path}') 

        process_depth = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        depth_folder_path = os.path.join(folder_path, 'depth_rcam_fish')
        if not os.path.exists(depth_folder_path):
            print("Depth folder not found")
            continue

        depths = sorted(os.listdir(depth_folder_path))
        print(len(depths))
        depth_batch_list = []
        for idx in range(len(depths)):
            if depths[idx].endswith('.png'):
                depth_path = os.path.join(depth_folder_path, depths[idx])
                # (1000, 1000)
                depth = Image.open(depth_path)

                depth_tensor = process_depth(depth)
                # extract the first three channels
                depth_tensor = depth_tensor[:3, :, :]
                
                depth_batch = depth_tensor.unsqueeze(0)
                print(depth_batch.shape)
                depth_batch_list.append(depth_batch)
        save_file_path = folder_path + '/depth_rcam_fish.pkl'
        with open(save_file_path, 'wb') as f:
            pickle.dump(depth_batch_list, f)

def load_preprocessed_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
'''
directory structure:
-main_folder_path-t1-depth, image, pose, imu
'''
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
        
        print(f'Processing: {folder_path}') 

        pose_file_path = os.path.join(folder_path, 'pose_lcam_front.txt')
        if not os.path.exists(pose_file_path):
            print("Pose file not found")
            continue

        with open(pose_file_path) as f:
            for line in f.readlines():
                values = line.strip().split()
                x, y, z = map(float, values[:3])
                positions.append(np.array([x, y, z]))
        print(positions[0], positions[-1])

        model_name = 'vit_base_patch16_224'
        model = timm.create_model(model_name, pretrained=True)
        model.eval()


        image_pkl_path = os.path.join(folder_path, 'image_lcam_fish.pkl')
        if not os.path.exists(image_pkl_path):
            print("Image folder not found")
            continue
        img_batch_list = load_preprocessed_data(image_pkl_path)
        depth_pkl_path = os.path.join(folder_path, 'depth_rcam_fish.pkl')
        if not os.path.exists(depth_pkl_path):
            print("Depth folder not found")
            continue
        depth_batch_list = load_preprocessed_data(depth_pkl_path)

        positions = np.array(positions)  # Convert positions to a numpy array

        for idx in range(len(img_batch_list)):
            with torch.no_grad():
                img_embedding = model.forward_features(img_batch_list[idx]) # last_hidden_state of the ViT model
                # (197,768)
                img_embedding = img_embedding.reshape(-1).numpy() # shape: (151296,)
                depth_embedding = model.forward_features(depth_batch_list[idx]) # last_hidden_state of the ViT model
                depth_embedding = depth_embedding.reshape(-1).numpy() # shape: (151296,)

            state = np.hstack((img_embedding,depth_embedding, positions[idx]))  # Stack the embeddings and positions horizontally
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