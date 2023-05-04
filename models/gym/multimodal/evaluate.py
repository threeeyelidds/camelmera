import gym
import numpy as np
from d3rlpy.algos import CQL

import sys
sys.path.append('/home/ubuntu/camelmera/models/gym/Q_learning')

from custom_env import AirSimDroneEnv

def evaluate_trained_model(env, model, episodes=10):
    rewards = []
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = model.predict([state])[0]
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward}")
    print(f"Average reward: {np.mean(rewards)}")

# create gym-like environment
env = AirSimDroneEnv(ip_address="127.0.0.1", step_length=0.5, image_shape=(224, 224, 3))

# load the pre-trained CQL model
cql = CQL.load("trained_cql.zip")

# evaluate the trained model on the custom environment
evaluate_trained_model(env, cql)
