from collections import deque
import torch 
import numpy as np


def train_decision_transformer_agent(agent, env, num_episodes=250, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.99, sequence_length=64):
    epsilon = epsilon_start
    rewards = []
    losses = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        episode_losses = []
        episode_steps = []

        action_history = []
        reward_history = []

        while not done:
            action = agent.act(state, action_history, reward_history)
            next_state, reward, done, _ = env.step(action)
            
            action_history.append(action)
            reward_history.append(reward)

            transition = (torch.tensor(np.array(state['position']), dtype=torch.float32).unsqueeze(0),
                          state['image'].unsqueeze(0),
                          torch.tensor(np.array([action]), dtype=torch.float32).unsqueeze(1),
                          torch.tensor(np.array([reward]), dtype=torch.float32).unsqueeze(1),
                          torch.tensor(np.array([done]), dtype=torch.float32).unsqueeze(1))

            


            episode_steps.append(transition)
            state = next_state
            total_reward += reward

            if done and len(episode_steps) >= sequence_length:
                experiences = list(zip(*episode_steps))
                loss = agent.learn(experiences)
                episode_losses.append(loss)

        epsilon = max(epsilon_end, epsilon_decay * epsilon)
        rewards.append(total_reward)
        losses.append(np.mean(episode_losses))
        print(f"Episode {episode + 1}/{num_episodes} - Reward: {total_reward} - Loss: {np.mean(episode_losses)}")

    return rewards, losses
