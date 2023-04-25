import torch
import torch.nn as nn
import torch.optim as optim
from custom_env import AirSimDroneEnv
from DDPG import Actor, Critic
import os
import pickle
import glob
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ip_address = '127.0.0.1'
step_length = 1
image_shape = (256, 256, 3)
env = AirSimDroneEnv(ip_address, step_length, image_shape)


BATCH_SIZE = 128
GAMMA = 0.99 # Reward decay factor
LR_ACTOR = 1e-4
LR_CRITIC = 1e-4

# Parameters used to initialize Acotr and Critic
n_actions = env.action_space.shape[0]
state = env.reset()
embedding_size = state["embedding"].shape[0]

# Define Actor and Critic and Optimizer
actor = Actor(embedding_size, n_actions).to(device)
critic = Critic(embedding_size, n_actions).to(device)
actor_optimizer = optim.AdamW(actor.parameters(), lr=LR_ACTOR)
critic_optimizer = optim.AdamW(critic.parameters(), lr=LR_CRITIC)


def load_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
def get_data(saved_folder_path):
    dataset_paths = glob.glob(os.path.join(saved_folder_path, 'dataset_*.pkl'))
    consolidated_dataset = {'observations': [], 'actions': [], 'rewards': []}

    if dataset_paths:
        print('Loading data from files...')
        for path in sorted(dataset_paths):
            try:
                data = load_data(path)
                consolidated_dataset['observations'].extend(data['observations'])
                consolidated_dataset['actions'].extend(data['actions'])
                consolidated_dataset['rewards'].extend(data['rewards'])
            except EOFError:
                print(f"Error loading data from file: {path}")
    else:
        print('file not found.')
    return consolidated_dataset


saved_folder_path = 'C:\\Users\\Tianyi\\Desktop\\11777\\preprocessed_all_data_easy'
data_file = os.path.join(saved_folder_path, 'dataset_0.pkl')


def select_action(state, goal_embedding):
    state_embedding = torch.tensor(state["embedding"], device=device).float().unsqueeze(0)
    combined_embedding = torch.cat((state_embedding, goal_embedding), dim=-1)
    with torch.no_grad():
        return actor(combined_embedding).max(1)[1].view(1, 1)


def optimize_model(batch):
    state_batch = torch.stack([torch.tensor(d['state']["embedding"]) for d in batch]).float().to(device)
    goal_batch = torch.stack([torch.tensor(d['goal']["embedding"]) for d in batch]).float().to(device)
    combined_state_goal_batch = torch.cat((state_batch, goal_batch), dim=-1)
    
    action_batch = torch.stack([torch.tensor(d['action']) for d in batch]).long().to(device)
    reward_batch = torch.stack([torch.tensor(d['reward']) for d in batch]).float().to(device)

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
    expected_state_action_values = discounted_rewards[:, 0]
    criterion = nn.SmoothL1Loss()
    loss_critic = criterion(current_state_values, expected_state_action_values.unsqueeze(1))
    critic_optimizer.zero_grad()
    loss_critic.backward()
    torch.nn.utils.clip_grad_value_(critic.parameters(), 100)
    critic_optimizer.step()


    # Update actor
    loss_actor = -critic(combined_state_goal_batch, actor(combined_state_goal_batch)).mean()
    actor_optimizer.zero_grad()
    loss_actor.backward()
    torch.nn.utils.clip_grad_value_(actor.parameters(), 100)
    actor_optimizer.step()


    return loss_critic.item(), loss_actor.item()

# Training Loop
# The get_data will return a list of datasets. 
datasets = get_data(saved_folder_path)

observations = datasets['observations']
actions = datasets['actions']
rewards = datasets['rewards']

print("dataset length:", len(observations))
num_batches = len(observations) // BATCH_SIZE

wandb.init(project='DDPG')
print("number of batches:", num_batches)

for i_batch in range(num_batches):
    start_idx = i_batch * BATCH_SIZE
    end_idx = start_idx + BATCH_SIZE

    batch_observations = observations[start_idx:end_idx]
    batch_actions = actions[start_idx:end_idx]
    batch_rewards = rewards[start_idx:end_idx]

    batch = []
    for obs, act, rew in zip(batch_observations, batch_actions, batch_rewards):
        batch.append({'state': {'embedding': obs}, 'action': act, 'reward': rew})

    loss_critic, loss_actor = optimize_model(batch)

    # Log loss and episode duration in WandB
    wandb.log({"Batch Duration": BATCH_SIZE, "Critic Loss": loss_critic, "Actor Loss": loss_actor})

    # Print losses in the terminal
    print(f"Batch: {i_batch+1}/{num_batches}, Critic Loss: {loss_critic:.6f}, Actor Loss: {loss_actor:.6f}")

print('Training Complete')
