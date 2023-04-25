import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from DDPG import Actor, Critic
import os
import wandb
from transformers import ViTMAEConfig
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append( os.path.join(parent_dir, "multimodal")  )

from custom_models.CustomViT import CustomViT
from custom_models.CustomViTMAE import CustomViTMAE
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEModel
from tem_dataloader import MultimodalDatasetPerTrajectory
from torch.utils.data import DataLoader

import wandb
wandb.login() 

trained_model_name = "multimodal_RL"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
environment_name = 'AbandonedCableExposure'
model_bin_dir = '/home/ubuntu/camelmera/weights'
environemnt_directory = f'/mnt/data/tartanairv2filtered/{environment_name}/Data_easy'

# Initialize a new CustomViTMAE model
model_name = "facebook/vit-mae-base"

vit_config = ViTMAEConfig.from_pretrained(model_name)
vit_config.output_hidden_states=True
vit_model = CustomViT.from_pretrained(model_name,config=vit_config)

model_name = "facebook/vit-mae-base"

config = ViTMAEConfig.from_pretrained(model_name)
config.output_hidden_states=True

# load from pretrained model and replace the original encoder with custom encoder
custom_model = CustomViTMAE.from_pretrained("facebook/vit-mae-base",config=config)
custom_model.vit = vit_model
custom_model.eval()

# preprocess_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
preprocess_device = "cpu"
print(preprocess_device)
custom_model.to(preprocess_device)

# Load the state_dict from the saved model
# state_dict = torch.load(os.path.join(model_bin_dir, "pytorch_model.bin"), map_location=torch.device('cpu'))
state_dict = torch.load(os.path.join(model_bin_dir, "pytorch_model.bin"))

# Apply the state_dict to the custom_model
custom_model.load_state_dict(state_dict)

# create Unimodel ViT
unimodal_model_name = "facebook/vit-mae-base"
unimodal_vit_config = ViTMAEConfig.from_pretrained(unimodal_model_name)
unimodal_vit_config.output_hidden_states=True
unimodal_vit_model = ViTMAEModel.from_pretrained(unimodal_model_name, config=unimodal_vit_config)
unimodal_vit_model.eval()
unimodal_vit_model.to(preprocess_device)

BATCH_SIZE = 32
GAMMA = 0.99  # Reward decay factor
LR_ACTOR = 1e-4
LR_CRITIC = 1e-4

# Define Actor and Critic and Optimizer
action_size = 7  # Assuming you have 4 actions in your action space
state_size = 768
goal_size = 768

combine_state_goal_size = state_size + goal_size  # Assuming your state embeddings have a size of 128

actor = Actor(combine_state_goal_size, action_size).to(device)
critic = Critic(combine_state_goal_size, action_size).to(device)

# Set the path where your trained models are saved
model_save_path = os.path.join(trained_model_name, 'trained_models')
actor_model_path = os.path.join(model_save_path, 'actor.pth')
critic_model_path = os.path.join(model_save_path, 'critic.pth')

if os.path.exists(actor_model_path) and os.path.exists(critic_model_path):
    # Load the saved state dictionaries into the models
    actor.load_state_dict(torch.load(actor_model_path, map_location=device))
    critic.load_state_dict(torch.load(critic_model_path, map_location=device))

actor_optimizer = optim.AdamW(actor.parameters(), lr=LR_ACTOR)
critic_optimizer = optim.AdamW(critic.parameters(), lr=LR_CRITIC)

def optimize_model(batch):
    state_batch = torch.stack([d['state']["embedding"].clone().detach() for d in batch]).float().to(device)
    goal_batch = torch.stack([d['goal']["embedding"].clone().detach() for d in batch]).float().to(device)
    combined_state_goal_batch = torch.cat((state_batch, goal_batch), dim=-1)

    action_batch = torch.stack([d['action'].clone().detach() for d in batch]).float().to(device)
    reward_batch = torch.stack([d['reward'].clone().detach() for d in batch]).float().to(device)

    # print(f"state_batch.dtype {state_batch.dtype}")
    # print(f"goal_batch.dtype {goal_batch.dtype}")
    # print(f"action_batch.dtype {action_batch.dtype}")
    # print(f"reward_batch.dtype {reward_batch.dtype}")
    # print(f"combined_state_goal_batch.dtype {combined_state_goal_batch.dtype}")

    # Update critic
    current_state_values = critic(combined_state_goal_batch, action_batch) # Calculate current Q-values using critic network
    # We are using ground truth rewards for the expected Q value
    discounted_rewards = torch.zeros_like(reward_batch)
    discounted_rewards[:, -1] = reward_batch[:, -1]
    for t in reversed(range(reward_batch.size(1) - 1)):
        discounted_rewards[:, t] = reward_batch[:, t] + GAMMA * discounted_rewards[:, t + 1]
    # If we are not using ground truth, we will call critic again for the expected Q functions. 
    # next_state_values = target_critic(combined_next_state_goal_batch, actions_next) # Use target_critic to get next Q-values
    # expected_state_action_values = (next_state_values * GAMMA) + reward_batch[:, 0]
    expected_state_action_values = discounted_rewards
    criterion = nn.SmoothL1Loss()
    # print(f"current_state_values.dtype {current_state_values.dtype}")
    # print(f"expected_state_action_values.dtype {expected_state_action_values.dtype}")

    loss_critic = criterion(current_state_values, expected_state_action_values)
    critic_optimizer.zero_grad()
    loss_critic.backward()
    torch.nn.utils.clip_grad_value_(critic.parameters(), 100)
    critic_optimizer.step()


    # Update actor
    actor_output = actor(combined_state_goal_batch)
    # print(f"actor_output.dtype {actor_output.dtype}")
    loss_actor = -critic(combined_state_goal_batch, actor_output).mean()
    actor_optimizer.zero_grad()
    loss_actor.backward()
    torch.nn.utils.clip_grad_value_(actor.parameters(), 100)
    actor_optimizer.step()


    return loss_critic.item(), loss_actor.item()

wandb.init(project="11777",name=trained_model_name+"_"+environment_name+time.strftime("%Y%m%d-%H%M%S"))
# Training Loop
for folder in os.listdir(environemnt_directory):
    trajectory_folder_path = os.path.join(environemnt_directory, folder)
    if not os.path.isdir(trajectory_folder_path):
        continue
    my_dataset = MultimodalDatasetPerTrajectory(trajectory_folder_path)
    train_dataloader = DataLoader(my_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    for index, raw_batch in enumerate(train_dataloader):
        RL_input_dictionary = {'merged_embeddings': [], 'goal': [], 'actions': [], 'rewards': []}
        pixel_values = raw_batch["pixel_values"].to(preprocess_device)
        pixel_values1 = raw_batch["pixel_values1"].to(preprocess_device)
        pixel_values2 = raw_batch["pixel_values2"].to(preprocess_device)
        pose_values = raw_batch["pose_values"].to(preprocess_device)
        merged_embedding = custom_model(pixel_values,pixel_values1,pixel_values2,noise=None)

        for element_index in range(pixel_values.shape[0]):
            RL_input_dictionary["merged_embeddings"].append(merged_embedding.hidden_states[element_index,0,:])

        for element_index_1 in range(pixel_values.shape[0] - 1):
            RL_input_dictionary["actions"].append(pose_values[element_index_1 + 1] - pose_values[element_index_1])
        RL_input_dictionary["actions"].append(torch.zeros_like(RL_input_dictionary["actions"][-1]))

        unimodal_outputs = unimodal_vit_model(pixel_values[-1,:,:,:].unsqueeze(0))
        RL_input_dictionary['goal'].append(unimodal_outputs.last_hidden_state[0,0,:])

        for element_index in range(pixel_values.shape[0]):
            image_embed = unimodal_vit_model(pixel_values[element_index,:,:,:].unsqueeze(0)).last_hidden_state[0,0,:]
            reward = -np.linalg.norm((image_embed - RL_input_dictionary['goal'][0]).detach().numpy())
            RL_input_dictionary["rewards"].append(torch.tensor(reward, dtype=torch.float32).unsqueeze(0))

        batch_embeddings = RL_input_dictionary['merged_embeddings']
        batch_actions = RL_input_dictionary['actions']
        batch_goal = RL_input_dictionary['goal']
        batch_reward = RL_input_dictionary['rewards']

        batch = []
        for emb, act, rwd in zip(batch_embeddings, batch_actions, batch_reward):
            # print(emb.shape)
            # print(act.shape)
            batch.append({'state': {'embedding': emb}, 'action': act, 'goal': {'embedding': batch_goal[0]}, 'reward': rwd})

        loss_critic, loss_actor = optimize_model(batch)

        # Log loss and episode duration in WandB
        wandb.log({"Batch Duration": BATCH_SIZE, "Critic Loss": loss_critic, "Actor Loss": loss_actor})

print('Training Complete')
wandb.finish()

# Set the path where you want to save your trained models
os.makedirs(model_save_path, exist_ok=True)

torch.save(actor.state_dict(), actor_model_path)
torch.save(critic.state_dict(), critic_model_path)

print(f'Trained models saved to {actor_model_path} and {critic_model_path}')
