import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from itertools import count
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = custom_env()

BATCH_SIZE = 128
GAMMA = 0.99 # Reward decay factor
LR = 1e-4

n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

def select_action(state):
    with torch.no_grad():
        return policy_net(state).max(1)[1].view(1, 1)

def optimize_model(batch):
    state_batch = torch.stack([torch.tensor(d['state']) for d in batch]).float().to(device)
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
