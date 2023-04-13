import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from itertools import count
import wandb
from custom_env import AirSimDroneEnv
from DQN import DQN
import os
import pickle
import glob


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Replace custom_env() with AirSimDroneEnv and pass the required parameters
ip_address = '127.0.0.1'
step_length = 1
image_shape = (84, 84, 1)
env = AirSimDroneEnv(ip_address, step_length, image_shape)


BATCH_SIZE = 128
GAMMA = 0.99 # Reward decay factor
LR = 1e-4

n_actions = env.action_space.n
state, info = env.reset()

embedding_size = state["embedding"].shape[0]

policy_net = DQN(embedding_size, n_actions).to(device)
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

def select_action(state):
    state_embedding = torch.tensor(state["embedding"], device=device).float().unsqueeze(0)
    with torch.no_grad():
        return policy_net(state_embedding).max(1)[1].view(1, 1)

def optimize_model(batch):
    state_batch = torch.stack([torch.tensor(d['state']["embedding"]) for d in batch]).float().to(device)
    action_batch = torch.stack([torch.tensor(d['action']) for d in batch]).long().to(device)
    reward_batch = torch.stack([torch.tensor(d['reward']) for d in batch]).float().to(device)

    state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values = torch.sum(reward_batch, dim=1)

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch[:, 0]

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    return loss.item()

def load_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
def get_data(main_folder_path, goal_position, saved_folder_path):
    dataset_paths = glob.glob(os.path.join(saved_folder_path, 'dataset_*.pkl'))

    if dataset_paths:
        print('Loading data from files...')
        datasets = [load_data(path) for path in sorted(dataset_paths)]
    else:
        print('file not found.')
    return datasets

goal_position = np.array([10, 10, 10])
saved_folder_path = '/home/tyz/Desktop/11_777'
data_file = os.path.join(saved_folder_path, 'preprocessed_all_data_easy.pkl')
main_folder_path = '/home/tyz/Desktop/11_777/Data_easy'


dataset = get_data(main_folder_path, goal_position, data_file)

num_episodes = len(dataset) // BATCH_SIZE
wandb.init(project='your_project_name')

for i_episode in range(num_episodes):
    start_idx = i_episode * BATCH_SIZE
    end_idx = start_idx + BATCH_SIZE

    batch = dataset[start_idx:end_idx]
    loss = optimize_model(batch)

    # Log loss and episode duration in WandB
    wandb.log({"Episode Duration": BATCH_SIZE, "Loss": loss})

print('Training Complete')
