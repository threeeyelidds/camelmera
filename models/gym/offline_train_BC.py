import sys
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from transformers import ViTMAEConfig
import torch.nn.functional as F


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, "multimodal"))

from multimodal.custom_models.CustomViT import CustomViT
from multimodal.custom_models.CustomViTMAE import CustomViTMAE
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEModel
from multimodal.tem_dataloader import MultimodalDatasetPerTrajectory

# Define BCModel
class BCModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(BCModel, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

trained_model_name = "multimodal_BC"

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

preprocess_device = "cpu"
print(preprocess_device)
custom_model.to(preprocess_device)

state_dict = torch.load(os.path.join(model_bin_dir, "pytorch_model.bin"))

custom_model.load_state_dict(state_dict)

unimodal_model_name = "facebook/vit-mae-base"
unimodal_vit_config = ViTMAEConfig.from_pretrained(unimodal_model_name)
unimodal_vit_config.output_hidden_states=True
unimodal_vit_model = ViTMAEModel.from_pretrained(unimodal_model_name, config=unimodal_vit_config)
unimodal_vit_model.eval()
unimodal_vit_model.to(preprocess_device)

action_size = 7
state_size = 768
goal_size = 768

combine_state_goal_size = state_size + goal_size

bc_model = BCModel(combine_state_goal_size, action_size).to(device)
bc_optimizer = optim.AdamW(bc_model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

def optimize_model(batch):
    state_batch = torch.stack([d['state']["embedding"].clone().detach() for d in batch]).float().to(device)
    goal_batch = torch.stack([d['goal']["embedding"].clone().detach() for d in batch]).float().to(device)
    combined_state_goal_batch = torch.cat((state_batch, goal_batch), dim=-1)

    expert_action_batch = torch.stack([d['action'].clone().detach() for d in batch]).float().to(device)

    predicted_action_batch = bc_model(combined_state_goal_batch)
    loss = criterion(predicted_action_batch, expert_action_batch)

    bc_optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(bc_model.parameters(), 100)
    bc_optimizer.step()

    return loss.item()

wandb.init(project="11777",name=trained_model_name+"_"+environment_name+time.strftime("%Y%m%d-%H%M%S"))

for folder in os.listdir(environemnt_directory):
    trajectory_folder_path = os.path.join(environemnt_directory, folder)
    if not os.path.isdir(trajectory_folder_path):
        continue
    my_dataset = MultimodalDatasetPerTrajectory(trajectory_folder_path)
    train_dataloader = DataLoader(my_dataset, batch_size=32, shuffle=False)
    
    for index, raw_batch in enumerate(train_dataloader):
        RL_input_dictionary = {'merged_embeddings': [], 'goal': [], 'actions': [], 'rewards': []}
        pixel_values = raw_batch["pixel_values"].to(preprocess_device)
        pixel_values1 = raw_batch["pixel_values1"].to(preprocess_device)
        pixel_values2 = raw_batch["pixel_values2"].to(preprocess_device)
        pose_values = raw_batch["pose_values"].to(preprocess_device)
        merged_embedding = custom_model(pixel_values, pixel_values1, pixel_values2, noise=None)

        for element_index in range(pixel_values.shape[0]):
            RL_input_dictionary["merged_embeddings"].append(merged_embedding.hidden_states[element_index,0,:])

        for element_index_1 in range(pixel_values.shape[0] - 1):
            RL_input_dictionary["actions"].append(pose_values[element_index_1 + 1] - pose_values[element_index_1])
        RL_input_dictionary["actions"].append(torch.zeros_like(RL_input_dictionary["actions"][-1]))

        unimodal_outputs = unimodal_vit_model(pixel_values[-1,:,:,:].unsqueeze(0))
        RL_input_dictionary['goal'].append(unimodal_outputs.last_hidden_state[0,0,:])

        batch_embeddings = RL_input_dictionary['merged_embeddings']
        batch_actions = RL_input_dictionary['actions']
        batch_goal = RL_input_dictionary['goal']

        batch = []
        for emb, act in zip(batch_embeddings, batch_actions):
            batch.append({'state': {'embedding': emb}, 'action': act, 'goal': {'embedding': batch_goal[0]}})

        loss = optimize_model(batch)
        wandb.log({"Batch Duration": 32, "Loss": loss})

print('Training Complete')
wandb.finish()

model_save_path = os.path.join(trained_model_name, 'trained_models')
bc_model_path = os.path.join(model_save_path, 'bc_model.pth')
os.makedirs(model_save_path, exist_ok=True)

torch.save(bc_model.state_dict(), bc_model_path)
print(f'Trained model saved to {bc_model_path}')

