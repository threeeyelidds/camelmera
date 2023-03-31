import gym
import numpy as np
import torch
import wandb

import argparse
import pickle
import random
import sys

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.training.seq_trainer import SequenceTrainer

import os
from PIL import Image
import timm
from torchvision import transforms


from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

from gym.envs.registration import register

# Graph imports
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
# from decision_transformer.envs.custom_env import AirSimDroneEnv

register(
    id="airsim-drone-sample-v0", entry_point="decision_transformer.envs.custom_env:AirSimDroneEnv",
)


def save_preprocessed_data(dataset, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f'Saved preprocessed data to {file_path}')


def load_preprocessed_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def get_preprocessed_data(main_folder_path, goal_position, preprocessed_data_file):
    if os.path.exists(preprocessed_data_file):
        print('Loading preprocessed data from file...')
        datasets = load_preprocessed_data(preprocessed_data_file)
    else:
        print('Preprocessing data...')
        datasets = load_data(main_folder_path, goal_position)
        print('Saving preprocessed data to file...')
        save_preprocessed_data(datasets, preprocessed_data_file)

    return datasets

def load_data(main_folder_path, goal_position):
    all_datasets = []

    for folder in os.listdir(main_folder_path):
        folder_path = os.path.join(main_folder_path, folder)

        if not os.path.isdir(folder_path):
            continue

        states = []
        actions = []
        rewards = []
        positions = []
        
        print(f'Processing folder: {folder_path}') 

        pose_file_path = os.path.join(folder_path, 'pose_lcam_front.txt')
        if not os.path.exists(pose_file_path):
            continue

        with open(pose_file_path) as f:
            for line in f.readlines():
                values = line.strip().split()
                x, y, z = map(float, values[:3])
                positions.append(np.array([x, y, z]))

        model_name = 'vit_base_patch16_224'
        model = timm.create_model(model_name, pretrained=True)
        model.eval()

        preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        image_folder_path = os.path.join(folder_path, 'image_lcam_front')
        if not os.path.exists(image_folder_path):
            continue

        positions = np.array(positions)  # Convert positions to a numpy array

        for idx, img_name in enumerate(sorted(os.listdir(image_folder_path))):
            if img_name.endswith('.png'):
                img_path = os.path.join(image_folder_path, img_name)
                img = Image.open(img_path)

                input_tensor = preprocess(img)
                input_batch = input_tensor.unsqueeze(0)
                with torch.no_grad():
                    embedding = model.forward_features(input_batch) # Delete the classification result

                embedding = embedding.squeeze().reshape(-1).numpy() # Reshape the embedding to a 1D array

                
                state = np.hstack((embedding, positions[idx]))  # Stack the embeddings and positions horizontally
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




    
def get_dimensions(dataset):
    state_example, action_example = dataset['observations'][0], dataset['actions'][0]
    state_dim = state_example.shape[0]
    action_dim = action_example.shape[0]

    return state_dim, action_dim



def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
        
    return discount_cumsum

main_folder_path = '/media/jeffrey/2TB HHD/AbandonedCableExposure/Data_easy'
goal_position = np.array([10, 10, 10]) # One point in P000 Easy trajectory
saved_folder_path = '/media/jeffrey/2TB HHD/camelmera'
preprocessed_data_file = os.path.join(saved_folder_path, 'preprocessed_imgae_depth_pos_imu_v006.pkl')
preprocessed_data_file1 = os.path.join(saved_folder_path, 'preprocessed_imgae_depth_pos_imu_v001.pkl')
preprocessed_data_file2 = os.path.join(saved_folder_path, 'preprocessed_imgae_depth_pos_imu_v002.pkl')
# env = DummyVecEnv(
#     [
#         lambda: Monitor(
#             gym.make(
#                 'airsim-drone-sample-v0',
#                 ip_address="127.0.0.1",
#                 step_length=0.25,
#                 image_shape=(84, 84, 1),
#             )
#         )
#     ]
# )

# Wrap env as VecTransposeImage to allow SB to handle frame observations
# env = VecTransposeImage(env)


def experiment(
        exp_prefix,
        variant,
):
    
    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)

    max_ep_len = 1000
    env_targets = [5000, 2500]
    scale = 1000.

    goal_position = np.array([10, 10, 10])  

    # trajectories = get_preprocessed_data(main_folder_path, goal_position, preprocessed_data_file)

    # Get trajectory from a single dictionary

    

    trajectory6 = get_preprocessed_data(main_folder_path, goal_position, preprocessed_data_file)

    trajectory1 = get_preprocessed_data(main_folder_path, goal_position, preprocessed_data_file1)
    trajectory2 = get_preprocessed_data(main_folder_path, goal_position, preprocessed_data_file2)

    def normalize_data(data):
        normalized_data = {}
        
        observation = data[0]['observations']
        action = data[0]['actions']
        reward = data[0]['rewards']


        normalized_observation = (observation-np.mean(observation,axis=0,keepdims=True))/np.std(observation,axis=0,keepdims=True)
        normalized_action = (action-np.mean(action,axis=0,keepdims=True))/np.std(action,axis=0,keepdims=True)
        normalized_reward = (reward-np.mean(reward,axis=0,keepdims=True))/np.std(reward,axis=0,keepdims=True)

        normalized_data = {'observations': normalized_observation, 'actions': normalized_action, 'rewards': normalized_reward}
        
        return normalized_data

   
    

    trajectories = [normalize_data(trajectory1), normalize_data(trajectory2), normalize_data(trajectory6)]
    
    print("Number of trajs", len(trajectories))
    print("number of actions in trajectories", len(trajectories[0]['actions']), len(trajectories[1]['actions']), len(trajectories[2]['actions']))
    print("number of rewards in trajectories", len(trajectories[0]['rewards']), len(trajectories[1]['rewards']), len(trajectories[2]['rewards']))
    print("the shape of observations in trajectories", trajectories[0]['observations'].shape, trajectories[1]['observations'].shape, trajectories[2]['observations'].shape)
    if not trajectories:
        print("No trajectories found in the given folder.")
    else:
        state_dim, act_dim = get_dimensions(trajectories[0])

    for i, traj in enumerate(trajectories):
        start_position = traj['observations'][0][-3:]  # Get the last three elements of the first observation
        end_position = traj['observations'][-1][-3:]   # Get the last three elements of the last observation
        print(f"Trajectory {i + 1}: Start position = {start_position}, End position = {end_position}")

    # mode = variant.get('mode', 'normal')

    states, traj_lens, returns = [], [], []
    for path in trajectories:
        # if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
        #     path['rewards'][-1] = path['rewards'].sum()
        #     path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(np.sum(path['rewards']))
    traj_lens, returns = np.array(traj_lens), np.array(returns)


    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)
    # Print the first 100 rewards from the dataset
    # num_rewards_to_print = 10
    # rewards_printed = 0
    # for path in trajectories:
    #     for reward in path['rewards']:
    #         if rewards_printed < num_rewards_to_print:
    #             print(f"Reward {rewards_printed + 1}: {reward}")
    #             rewards_printed += 1
    #         else:
    #             break
        # if rewards_printed >= num_rewards_to_print:
        #     break

    # Print the sum of rewards and trajectory length for each trajectory
    for i, path in enumerate(trajectories):
        print(f"Trajectory {i + 1}: Sum of rewards = {sum(path['rewards'])}, Trajectory length = {len(path['rewards'])}")

    print("Current Directory", os.getcwd())
    print('=' * 50)
    print(f'Starting new experiment')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=512, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            #print ("trajectory reward shape: ", traj['rewards'].shape[0])
            # print (traj['rewards'].shape[0])
            si = random.randint(0, traj['rewards'].shape[0] - K)
            # si = 0
            # print('actions',traj['actions'].shape[0] - 1)
            # print('observations',traj['observations'].shape[0] - 1)
            # print('rewards',traj['rewards'].shape[0] - 1)
            # print('si',si)

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            
            # print (traj['actions'][si:si + max_len].reshape(1, -1, act_dim).shape)
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            # if 'terminals' in traj:
            #     d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            # else:
            #     d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff

            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]

            # for i, arr in enumerate(a):
            #     print(f"Array at batch_number {i} has shape {arr.shape}")

            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            # d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        # d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask
    
    model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
        
    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    ret, length = evaluate_episode_rtg(
                        state_dim,
                        act_dim,
                        model,
                        max_ep_len=max_ep_len,
                        scale=scale,
                        target_return=target_rew/scale,
                        # mode=mode,
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device,
                        )
                returns.append(ret)
                lengths.append(length)
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
            }
        return fn



    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2)/torch.mean(a**2),
            # eval_fns= [eval_episodes(tar) for tar in env_targets],
        )

    # # Generate a list of initial states from the dataset
    # initial_states = [traj['observations'][0] for traj in trajectories]

    # # Generate model-predicted trajectories
    # num_steps = 50  # You can adjust the number of steps in the predicted trajectories
    # model_trajectories = [generate_model_trajectory(model, init_state, num_steps, state_mean, state_std) for init_state in initial_states]

    # # Extract positions from dataset trajectories
    # for traj in trajectories:
    #     traj['positions'] = traj['observations'][:, -3:]  # Assuming the last three elements of the state represent the position (x, y, z)

    # # Visualize the trajectories
    # visualize_trajectories(trajectories, model_trajectories)

    run = wandb.init(project='camelmera', config=variant)

    # Train the model using the trainer.train method
    print("Starting training...")
    for iter in range(variant['max_iters']):
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True)
        # print("Iteration:", iter+1, "Loss:", outputs['loss'])
        if log_to_wandb:
            wandb.log(outputs)


    torch.save(model.state_dict(), "trained_model_image_depth_imu_pos.pt")
    # Load the saved model
    loaded_model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
    current_directory = os.getcwd()
    print("Current working directory:", current_directory)    

    loaded_model.load_state_dict(torch.load('trained_model_image_depth_imu_pos.pt'))
    loaded_model.to(device=device)
    print ("model loaded")
    # Evaluate the loaded model using the modified evaluation function
    # for target_rew in env_targets:
    #     eval_fn = eval_episodes(target_rew)
    #     eval_results = eval_fn(loaded_model)
    #     print(f"Evaluation results for target {target_rew}: {eval_results}")

    # Visualize the model performance
    # visualize_model_performance(loaded_model, trajectories, goal_position, device)

    start_position = trajectories[0]['observations'][0][-3:]
    
    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            # group=group_name,
            project='decision-transformer',
            config=variant
        )
        wandb.watch(model)  # wandb has some bug

    run.finish()

    def visualize_performance(model, trajectories, start_position, goal_position):
        model.eval()

        training_positions = []
        testing_positions = []

        for traj in trajectories:
            training_positions.append(traj["observations"][:, -3:])  # Extract position information from the last 3 elements of state

            # Prepare input for the model
            states, actions, rewards, dones, rtg, timesteps, attention_mask = get_batch(256,K)
            action_target = torch.clone(actions)

            with torch.no_grad():
                _, action_preds, _ = model.forward(states, actions, rewards, rtg[:, :-1], timesteps, attention_mask = attention_mask)
                # act_dim = action_preds.shape[2]
                action_preds = action_preds[0,:,:]
                print(action_preds.shape)
                # print(action_preds[:,j,:])

            # Calculate testing positions from predicted actions
            testing_positions.append(traj["observations"][0, -3:] + np.cumsum(action_preds.cpu().numpy(), axis=0))

        testing_positions = np.concatenate(testing_positions, axis=0)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Color code for each training trajectory
        colors = cm.rainbow(np.linspace(0, 1, len(trajectories)))

        # Plot training trajectories
        for traj_positions, color in zip(training_positions, colors):
            ax.scatter(traj_positions[:, 0], traj_positions[:, 1], traj_positions[:, 2], c='red', alpha=0.1, marker='o', s=2)

        # Plot testing trajectories
        ax.scatter(testing_positions[:, 0], testing_positions[:, 1], testing_positions[:, 2], c='blue', alpha=0.1, marker='o', s=6)
        # Plot start and goal positions
        ax.scatter(start_position[0], start_position[1], start_position[2], c='green', marker='s', label='Start', s=100)
        ax.scatter(goal_position[0], goal_position[1], goal_position[2], c='purple', marker='*', label='Goal', s=100)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

        plt.show()



    visualize_performance(model, trajectories, start_position, goal_position)

# Example usage
# visualize_performance(model, trajectories, goal_position, start_position=None)


# Example usage
# visualize_performance(model, trajectories, goal_position, start_position=None)
# def visualize_model_performance(loaded_model, trajectories, goal_position, device='cuda'):
#     # Extract position information from trajectories
#     training_positions = [traj['observations'][:, -3:] for traj in trajectories]
    
#     # Create a 3D plot for the training trajectories
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     for positions in training_positions:
#         ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], alpha=0.1, color='blue')

#     # Visualize the start and goal positions
#     start_position = training_positions[0][0]
#     ax.scatter(start_position[0], start_position[1], start_position[2], c='green', marker='o', label='Start')
#     ax.scatter(goal_position[0], goal_position[1], goal_position[2], c='red', marker='o', label='Goal')

#     # Use the loaded model to predict actions for each state in the testing trajectory
#     testing_positions = [start_position]
#     state = torch.tensor(start_position, dtype=torch.float32).unsqueeze(0).to(device)
#     for _ in range(1000):  # 1000 is the maximum trajectory length
#         with torch.no_grad():
#             action = loaded_model(state).cpu().numpy()[0]
#         new_position = testing_positions[-1] + action
#         testing_positions.append(new_position)
#         state = torch.tensor(new_position, dtype=torch.float32).unsqueeze(0).to(device)

#     # Plot the testing trajectory
#     testing_positions = np.array(testing_positions)
#     ax.plot(testing_positions[:, 0], testing_positions[:, 1], testing_positions[:, 2], color='purple', label='Testing')

#     # Set labels and show the plot
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.legend()
#     plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--env', type=str, default='hopper')
    # parser.add_argument('--dataset', type=str, default='medium')  # medium, medium-replay, medium-expert, expert
    # parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=100)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=128)
    # parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    
    args = parser.parse_args()

    experiment('gym-experiment', variant = {key: value for key, value in vars(args).items() if key != "model_type"}
)
