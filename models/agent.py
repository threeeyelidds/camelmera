# import numpy as np              # For numerical operations
import torch                    # For creating the neural network
# import torch.nn as nn           # For neural network layers and functions
# import torch.optim as optim     # For optimization algorithms
# from PIL import Image           # For image processing


from environments import RobotEnv
from CQL_agent import CQLAgent
from utils import load_image_files, plot_rewards_losses
from train import train_cql_agent
from Decision_Transformer_agent import DecisionTransformerAgent
from train_decision_transformer import train_decision_transformer_agent


# Code Structure: 

# 1.Load the .txt file containing the robot's xyz positions
# 2.Load the image file containing the robot's environment
# 3.Define the environment by creating a custom class that inherits from gym.Env
# 4.Implement required methods: __init__, step, reset, and render
# 5.Define the state as the current xyz position, desired position, and image
# 6.Define the action space as discrete with 6 possible actions (±x, ±y, ±z)
# 7.Define the reward as the negative Euclidean distance between the current position and desired position
# 8.Implement a PyTorch reinforcement learning agent
# 9.Train the agent using an appropriate algorithm

# main code for CQL
# # Load data
# print ("Loading image files")
# image_folder = "/media/jeffrey/2TB HHD/Camelmera/models/image_lcam_front"  
# image_files = load_image_files(image_folder, file_ext=".png")

# print ("Loading trajectory information")
# filename = "/media/jeffrey/2TB HHD/Camelmera/models/Position_000.txt"
# desired_position = [9.939322644295340581e+01, 6.723854706303634998e+01, -6.008911815873547724e+00]

# # Define the custom Robot environment class which inherits from gym.Env
# print ("Creating Environment")
# env = RobotEnv(filename, image_files, desired_position)

# # Check for cuda
# if torch.cuda.is_available():
#     print("CUDA is available.")
# else:
#     print("CUDA is not available.")

# state_dim = env.observation_space['position'].shape[0]
# action_dim = env.action_space.n
# hidden_size = 256

# print ("Creating CQL Agent")

# device = "cuda" if torch.cuda.is_available() else "cpu"
# agent = CQLAgent(state_size=state_dim, action_size=action_dim, device=device)

# print ("Start Trainning")
# rewards, losses = train_cql_agent(agent, env)

# print ("Plotting")
# plot_rewards_losses(rewards, losses)


# Main code for decision transformer
print ("Loading image files")
image_folder = "/media/jeffrey/2TB HHD/Camelmera/models/image_lcam_front"
image_files = load_image_files(image_folder, file_ext=".png")

print ("Loading trajectory information")
filename = "/media/jeffrey/2TB HHD/Camelmera/models/Position_000.txt"
desired_position = [9.939322644295340581e+01, 6.723854706303634998e+01, -6.008911815873547724e+00]

print ("Creating Environment")
env = RobotEnv(filename, image_files, desired_position)

state_dim = env.observation_space['position'].shape[0]
action_dim = env.action_space.n
sequence_length = 64

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Creating Decision Transformer Agent")
agent = DecisionTransformerAgent(state_size=state_dim, action_size=action_dim,   sequence_length=sequence_length, device=device)

print ("Start Trainning")
rewards, losses = train_decision_transformer_agent(agent, env)

print ("Plotting")
plot_rewards_losses(rewards, losses)