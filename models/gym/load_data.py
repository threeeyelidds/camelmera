import os
import pickle
import numpy as np
import torch
import timm
from PIL import Image
from torchvision import transforms
import glob

'''
Define save_data function to save a dataset to a file using Pickle
Define load_data function to load a dataset from a file using Pickle
Define get_data function to load datasets from files if they exist, or process data and save it
Define process_data function to process all folders and return a list of datasets
Define process_and_save_data function to process folders, save each dataset to a file, and return a list of datasets
Define get_model function to create a pre-trained Vision Transformer model for image embeddings
Define get_preprocess function to create a pre-processing pipeline for image inputs
Define load_positions function to load position data from a text file
Define process_folder function to process a folder containing images and positions, returning a dataset with observations, actions, and rewards
Define get_embedding function to get the image embedding from the Vision Transformer model
Define get_dimensions function to get the dimensions of state and action vectors from a dataset
Define print_trajectory_info function to print information about trajectories
Define main function to run the script, load or process data, and print trajectory information

'''
def save_data(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f'Saved data to {file_path}')

def load_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def get_data(main_folder_path, goal_position, saved_folder_path):
    dataset_paths = glob.glob(os.path.join(saved_folder_path, 'dataset_*.pkl'))

    if dataset_paths:
        print('Loading data from files...')
        datasets = [load_data(path) for path in sorted(dataset_paths)]
    else:
        print('Processing data...')
        datasets = process_and_save_data(main_folder_path, goal_position, saved_folder_path)

    return datasets

def process_data(main_folder_path, goal_position):
    all_data = []
    model = get_model()
    preprocess = get_preprocess()

    for folder in os.listdir(main_folder_path):
        folder_path = os.path.join(main_folder_path, folder)
        if not os.path.isdir(folder_path): continue

        positions = load_positions(os.path.join(folder_path, 'pose_lcam_front.txt'))
        if positions is None: continue

        images_folder = os.path.join(folder_path, 'image_lcam_front')
        if not os.path.exists(images_folder): continue

        dataset = process_folder(images_folder, positions, model, preprocess, goal_position)
        all_data.append(dataset)

    return all_data


def process_and_save_data(main_folder_path, goal_position, saved_folder_path):
    os.makedirs(saved_folder_path, exist_ok=True)
    model = get_model()
    preprocess = get_preprocess()

    datasets = []
    for idx, folder in enumerate(os.listdir(main_folder_path)):
        folder_path = os.path.join(main_folder_path, folder)
        if not os.path.isdir(folder_path): continue

        positions = load_positions(os.path.join(folder_path, 'pose_lcam_front.txt'))
        if positions is None: continue

        images_folder = os.path.join(folder_path, 'image_lcam_front')
        if not os.path.exists(images_folder): continue

        print(f'Processing folder {idx + 1}: {folder_path}')
        dataset = process_folder(images_folder, positions, model, preprocess, goal_position)
        datasets.append(dataset)

        data_file = os.path.join(saved_folder_path, f'dataset_{idx}.pkl')
        save_data(dataset, data_file)

    return datasets


def get_model():
    model_name = 'vit_base_patch16_224'
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    return model

def get_preprocess():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def load_positions(file_path):
    if not os.path.exists(file_path): return None

    positions = []
    with open(file_path) as f:
        for line in f.readlines():
            x, y, z = map(float, line.strip().split()[:3])
            positions.append(np.array([x, y, z]))

    return np.array(positions)

def process_folder(folder_path, positions, model, preprocess, goal_position):
    states, actions, rewards = [], [], []

    for idx, img_name in enumerate(sorted(os.listdir(folder_path))):
        if not img_name.endswith('.png'): continue

        img_path = os.path.join(folder_path, img_name)
        img = Image.open(img_path)
        embedding = get_embedding(img, model, preprocess)
        state = np.hstack((embedding, positions[idx]))
        states.append(state)

        if idx > 0:
            action = positions[idx] - positions[idx - 1]
            actions.append(action)
            reward = -np.linalg.norm(positions[idx] - goal_position)
            rewards.append(reward)

    return {'observations': np.array(states), 'actions': np.array(actions), 'rewards': np.array(rewards)}

def get_embedding(img, model, preprocess):
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        embedding = model.forward_features(input_batch)
    return embedding.squeeze().reshape(-1).numpy()

def get_dimensions(dataset):
    state_example, action_example = dataset['observations'][0], dataset['actions'][0]
    return state_example.shape[0], action_example.shape[0]

def print_trajectory_info(trajectories):
    for i, traj in enumerate(trajectories):
        start_position = traj['observations'][0][-3:]
        end_position = traj['observations'][-1][-3:]
        print(f"Trajectory {i + 1}: Start position = {start_position}, End position = {end_position}")

def main():
    goal_position = np.array([10, 10, 10])
    saved_folder_path = '/home/tyz/Desktop/11_777'
    data_file = os.path.join(saved_folder_path, 'preprocessed_all_data_easy.pkl')
    main_folder_path = '/home/tyz/Desktop/11_777/Data_easy'

    trajectories = get_data(main_folder_path, goal_position, data_file)

    print("Number of trajectories:", len(trajectories))
    print_trajectory_info(trajectories)

    if not trajectories:
        print("No trajectories found in the given folder.")
        return

    state_dim, act_dim = get_dimensions(trajectories[0])

if __name__ == '__main__':
    main()