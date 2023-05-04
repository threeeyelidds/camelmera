import gym
import numpy as np
from d3rlpy.algos import CQL
import matplotlib.pyplot as plt


import sys
# sys.path.append('/home/ubuntu/camelmera/models/gym/Q_learning')

from custom_env import AirSimDroneEnv

def evaluate_trained_model(env, model, episodes=10):
    rewards = []
    trajectories = []
    final_distances = []
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_trajectory = []
        while not done:
            # action = model.predict([state])[0]
            # state, reward, done, _ = env.step(action)
            # episode_reward += reward
            action = model.predict([state])[0]
            next_state, reward, done, _ = env.step(action)
            episode_trajectory.append((state, action, reward))
            state = next_state
            episode_reward += reward
        rewards.append(episode_reward)
        trajectories.append(episode_trajectory)
        print(f"Episode {episode + 1}: Reward = {episode_reward}")

        # Calculate distance (L2 norm) between final position and goal_position
        # final_state = episode_trajectory[-1][0]
        # distance = np.linalg.norm(final_state[:2]-env.goal[:2])
        # final_distances.append(distance)

        # # if need to normalize: divide by total distance or total step 
        # print(f"Episode {episode + 1}: Final distance to goal = {distance}")


    print(f"Average reward: {np.mean(rewards)}")
    return trajectories, final_distances

def plot_trajectories(trajectories, goal_position):
    plt.figure()
    for idx, trajectory in enumerate(trajectories):
        states = [step[0][:2] for step in trajectory]
        x, y = zip(*states)
        plt.plot(x, y, label=f"Episode {idx + 1}")

    # Plot goal_position
    plt.plot(goal_position[0], goal_position[1], 'ro', label='Goal')
    plt.legend()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Trajectories')
    plt.show()

# load all observations
all_observations = np.load('all_observations.npy', mmap_mode='r')
# print(all_observations.shape)
goal_position = all_observations[500]
# print(all_observations[0].shape)
# print(f'this is goal shape: {np.zeros((768,)).shape}')

# ready to load
cql = CQL.from_json('params.json')
cql.load_model('model_4000.pt')

# initialize environment
# env = AirSimDroneEnv("127.0.0.1", -12, 17, np.zeros((768,)))
env = AirSimDroneEnv("127.0.0.1", -12, 17, goal_position)


# evaluate the trained model on the custom environment
trajectories, final_distances = evaluate_trained_model(env, cql)
print(f'These are the final distances for each episode: {final_distances}')


# Plot trajectories
plot_trajectories(trajectories, goal_position)
