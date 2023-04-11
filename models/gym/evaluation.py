
import numpy as np
import torch
from experiment import discount_cumsum
from decision_transformer.models.decision_transformer import DecisionTransformer
import os
import torch



def compute_reward(start_position, goal_position):
  start_pos = start_position.copy()
  goal_pos = goal_position.copy()
  return -np.linalg.norm(start_pos - goal_pos)

def normalize_data(data):
  normalized_data = {}
  
  observation = data[0]['observations']
  action = data[0]['actions']
  reward = data[0]['rewards']

  if contains_imu:
    pos_mean = np.mean(observation,axis=0,keepdims=True)[0,-243:-240]
    pos_std = np.std(observation,axis=0,keepdims=True)[0,-243:-240]
  else:
    pos_mean = np.mean(observation,axis=0,keepdims=True)[0,-3:]
    pos_std = np.std(observation,axis=0,keepdims=True)[0,-3:]
  normalized_observation = (observation-np.mean(observation,axis=0,keepdims=True))/np.std(observation,axis=0,keepdims=True)
  normalized_action = (action-np.mean(action,axis=0,keepdims=True))/np.std(action,axis=0,keepdims=True)
  normalized_reward = (reward-np.mean(reward,axis=0,keepdims=True))/np.std(reward,axis=0,keepdims=True)

  normalized_data = {'observations': normalized_observation, 'actions': normalized_action, 'rewards': normalized_reward}
  
  return normalized_data,pos_mean,pos_std



def evaluate(model, trajectory, goal_position, prediction_step=100):
  model.eval()
  
  if contains_imu:
    # Extract training positions from state
    training_positions = trajectory["observations"][:, 14184:14187].copy()
    # start position is the start of the trajectory
    start_position = trajectory["observations"][0,14184:14187].copy()
  else:
    # Extract training positions from state
    training_positions = trajectory["observations"][:, -3:].copy()
    # start position is the start of the trajectory
    start_position = trajectory["observations"][0,-3:].copy()

  testing_positions = []
  print("start position ", start_position)
  print("training start position", training_positions[0])
  print("goal position ", goal_position)
  
  # initialize
  # initialize state with the initial state of the trajectory
  states = trajectory["observations"][0].copy()
  states = torch.tensor(states.reshape(1,1,-1),dtype=torch.float32)
  # initialize action towards goal
  actions = torch.tensor((goal_position-start_position)*0.1,dtype=torch.float32).reshape(1,1,-1) # todo
  print("initial action", actions)
  # compute initial reward
  rewards = torch.tensor([compute_reward(start_position, goal_position)],dtype=torch.float32).reshape(1,-1)
  rtg = torch.tensor(discount_cumsum(rewards.clone(),1),dtype=torch.float32)
  timesteps = torch.tensor(np.arange(0, 1).reshape(1, -1),dtype=torch.int32)
  cur_pos = training_positions[0].copy()
  print("initial position", cur_pos)
  testing_positions.append(cur_pos.copy())
  # autoregressive prediction
  for i in range(prediction_step):
    # print("time step ", i)
    # print("actions ",actions,actions.dtype)
    # print("reward ",rewards.shape)
    # print("rtg ", rtg[:,-1])
    # print("timesteps", timesteps.shape)
    with torch.no_grad():
      # print("states ",states.shape,states.dtype)
      state_preds,action_preds, reward_preds = model.forward(states, actions, rewards, rtg[:,-1], timesteps)
      # print(reward_preds)
      next_state,next_action,_ = state_preds[:,-1,:],action_preds[:,-1,:],reward_preds[-1]
      # print(next_reward)
      # change position of next state
      pos_change = next_action.clone()
      cur_pos += pos_change.reshape(-1).cpu().numpy()
      testing_positions.append(cur_pos.copy())
      if contains_imu:
        # print("Position change ", next_state[:, :-240])
        next_state[:,14184:14187] = torch.tensor(cur_pos,dtype=torch.float32).reshape(1,-1)
      else:
        next_state[:,-3:] = torch.tensor(cur_pos,dtype=torch.float32).reshape(1,-1)
      # compute the reward yourself
      next_reward = torch.tensor([compute_reward(cur_pos, goal_position)],dtype=torch.float32).reshape(1,-1)
      states = torch.cat((states,next_state.reshape(1,1,-1)),dim=1)
      actions = torch.cat((actions,next_action.reshape(1,1,-1)),dim=1)
      rewards = torch.cat((rewards,next_reward.reshape(1,-1)),dim=1)
      rtg = torch.tensor(discount_cumsum(rewards.clone(),1),dtype=torch.float32)
      timesteps = torch.tensor(np.arange(0, states.shape[1]).reshape(1, -1))

  # (1,100,3)
  terminating_pos = testing_positions[-1].copy()
  testing_positions = np.array(testing_positions)
  distance_to_goal = np.linalg.norm(terminating_pos-goal_position)
  # print("actions ", actions)
  print("training positions shape ", training_positions.shape)
  print("testing positions shape ", testing_positions.shape)
  print("start position ", testing_positions[0])
  print("terminating position ", terminating_pos)
  print("goal position ", goal_position)
  print("distance to goal ", distance_to_goal)
  
  return training_positions,testing_positions,start_position,goal_position,distance_to_goal


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_performance(train_positions, test_positions, start_position, goal_position):
    train_positions = np.array(train_positions)
    test_positions = np.array(test_positions)
    start_position = list(start_position.squeeze())
    goal_position = list(goal_position.squeeze())
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the training positions
    x_train = train_positions[:, 0]
    y_train = train_positions[:, 1]
    z_train = train_positions[:, 2]
    ax.plot(x_train, y_train, z_train, color='blue', label='Training positions')
    
    # Plot the test positions
    x_test = test_positions[:, 0]
    y_test = test_positions[:, 1]
    z_test = test_positions[:, 2]
    ax.plot(x_test, y_test, z_test, color='green', label='Test positions')
    
    # Mark the start position
    ax.scatter(start_position[0], start_position[1], start_position[2], color='red', marker='o', label='Start position')
    
    # Mark the goal position
    ax.scatter(goal_position[0], goal_position[1], goal_position[2], color='purple', marker='*', label='Goal position')
    
    # Add axis labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    plt.show()

# change these params
model_name = "trained_model_image_depth_imu_pos"
file_path = "/media/jeffrey/2TB HHD/camelmera/preprocessed_image_depth_pos_imu_v002.pkl"

# unimodal: 7095, image+lidar/depth: 14187, image+lidar/depth+imu: 14187+240,
# check the state_dim yourself if there is any error

state_dim=14451
itertion=8
contains_imu=True


# Load the saved model
loaded_model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=3,
        max_length=100,
        max_ep_len=1000,
        hidden_size=128,
        n_layer=3,
        n_head=1,
        n_inner=4*128,
        activation_function='relu',
        n_positions=1024,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
    )

    
current_directory = os.getcwd()
print("Current working directory:", current_directory)    

loaded_model.load_state_dict(torch.load(f'{model_name}.pt'))
loaded_model.cpu()

import pickle
with open(file_path, 'rb') as f:
    trajectories = pickle.load(f)
trajectories,pos_mean,pos_std = normalize_data(trajectories)
goal_position=np.array([-2, -2, -8])
print("Goal_Position: ", goal_position)
print("Pos_mean", pos_mean)

  
# input: model, trajectory, goal position, prediction step
# return: train_trajectory (n,3), test_trajectory (n,3),start_pos, goal_pos, distance to goal

training_positions,testing_positions,start_position,goal_position,distance_to_goal=evaluate(loaded_model,trajectories,goal_position,40)

visualize_performance(training_positions,testing_positions,start_position,goal_position)

print("distance_to_goal", distance_to_goal)