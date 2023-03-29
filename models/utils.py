# Helper Functions
import os
import matplotlib.pyplot as plt
import torch

def load_image_files(folder_path, file_ext=".png"):
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(file_ext)]
    image_files.sort()
    return image_files


def plot_rewards_losses(rewards, losses):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward', color=color)
    ax1.plot(rewards, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Loss', color=color)  # we already handled the x-label with ax1
    ax2.plot(losses, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

import numpy as np

def compute_returns_to_go(rewards, dones, gamma=0.99):
    batch_size, seq_length = rewards.shape
    returns_to_go = torch.zeros_like(rewards, dtype=torch.float32)

    for t in reversed(range(seq_length)):
        if t == seq_length - 1:
            next_return = torch.zeros(batch_size, dtype=torch.float32).to(rewards.device)
        else:
            next_return = returns_to_go[:, t + 1]

        if dones is None:
            mask = 1
        else:
            mask = 1 - dones[:, t]

        returns_to_go[:, t] = rewards[:, t] + gamma * mask * next_return

    return returns_to_go
