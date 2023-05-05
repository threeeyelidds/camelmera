import gym
import numpy as np
from d3rlpy.algos import CQL,TD3PlusBC,AWAC,BCQ,BEAR
import matplotlib.pyplot as plt
import time

import sys
# sys.path.append('/home/ubuntu/camelmera/models/gym/Q_learning')

from custom_env import AirSimDroneEnv

# debug: change episodes back to 10 
# def evaluate_trained_model(env, model, episodes=10):
def evaluate_trained_model(env, model, episodes=3):

    rewards = []
    trajectories = []
    final_distances = []
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_trajectory = []
        count_steps = 0

        # add cumulative distance travelled if we want to measure the distance travelled against goal

        # remove count steps, just for debugging purpose
        while not done and count_steps < 10:
        # while not done:
            # action = model.predict([state])[0]
            # state, reward, done, _ = env.step(action)
            # episode_reward += reward
            action = model.predict([state])[0]
            print(f"this is action shape: {action.shape}")
            print(f"state shape: {state.shape}")
            print(f"current position shape: {env.position}")
            next_state, reward, done, _ = env.step(action)
            count_steps+=1
            episode_trajectory.append(env.position)
            # episode_trajectory.append((state, action, reward))
            state = next_state
            episode_reward += reward
            print(f'===========step: {count_steps}===============')
            # time.sleep(0.1)
            # print(f'========sleep: 0.1s=======')
        rewards.append(episode_reward)
        trajectories.append(episode_trajectory)
        print(f"Episode {episode + 1}: Reward = {episode_reward}")

        

        # Calculate distance (L2 norm) between final position and goal_position
        # final_state = episode_trajectory[-1][0]
        final_position = episode_trajectory[-1]
        # print(f'this is episode trajectory dimension: {(episode_trajectory[-1][0][0].shape)}')
        
        # print(f'this is goal shape: {env.goal.shape}')

        # do we use all 768 dimensions or just first 2?
        distance = np.linalg.norm(final_position-env.goal_xyz_coordinates)
        # distance = np.linalg.norm(final_state[0][:2]-env.goal[:2])
        final_distances.append(distance)

        # # if need to normalize: divide by total distance or total step 
        print(f"Episode {episode + 1}: Final distance to goal = {distance}")


    print(f"Average reward: {np.mean(rewards)}")
    return trajectories, final_distances

def plot_trajectories(trajectories, goal_position):
    fig = plt.figure()
    fig.add_subplot(projection='3d')
    # print(f'this is trajectories shape: {len(trajectories)}')
    for idx, trajectory in enumerate(trajectories):
        print(f"this is trajectory index {idx} and trajectory {trajectory}")
        
        x = []
        y = []
        z = []
        for step in trajectory:
            # print(f"this is step shape: {step.shape}")
            # fig.plot()
            # plt.plot(step, label=f"episode: {idx+1}")
            # plt.plot(step[0], step[1], step[2], label=f"episode: {idx+1}")
            x.append(step[0])
            y.append(step[1])
            z.append(step[2])

            # each step is (state, action, reward)
        #     print(f"these are the step size: {step[0].shape}")
        # states = [step[0][:2] for step in trajectory]
        # print(f"these are states dimensions: {len(states)}")
        # print(f"these are current states: {states[0].shape}")
        # print(f"these are current states: {states[1].shape}")
        # # x, y = zip(states)\
        plt.plot(x, y, z, label=f"Episode {idx + 1}")
        # plt.plot(states[0][0], states[1][0], label=f"Episode {idx + 1}")

        # plt.plot(x, y, label=f"Episode {idx + 1}")

    # Plot goal_position
    print(f"this is scalar for goal position: {goal_position}")
    plt.plot(goal_position[0], goal_position[1], goal_position[2], 'ro', label='Goal')
    plt.legend()
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    plt.title('Trajectories')
    plt.show()

# load all observations
all_observations = np.load('all_observations.npy', mmap_mode='r')
# print(all_observations.shape)
goal_position = all_observations[0]
goal_coordinates = np.array([1.830941200256347656e+01 ,-9.911462664604187012e-01, -2.512080669403076172e+00])
# print(f'goal shape: {all_observations[0].shape}')
# print(f'this is goal shape: {np.zeros((768,)).shape}')
print(f"this is goal coordinates: {goal_coordinates}")
# ready to load
cql = BEAR.from_json('BEAR1/params.json')
cql.load_model('BEAR1/model.pt')

# initialize environment
# env = AirSimDroneEnv("127.0.0.1", -12, 17, np.zeros((768,)))
# env = AirSimDroneEnv("127.0.0.1", -12, 17, goal_position)
env = AirSimDroneEnv("127.0.0.1", -12, 17, goal_position)



# evaluate the trained model on the custom environment
trajectories, final_distances = evaluate_trained_model(env, cql)
print(f'These are the final trajectories for each episode: {trajectories}')
print(f'These are the final distances for each episode: {final_distances}')


# Plot trajectories
# for trajectory in trajectories:
plot_trajectories(trajectories, goal_coordinates)
