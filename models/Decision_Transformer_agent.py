import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from decision_transformer.models.decision_transformer import DecisionTransformer
import torch.nn.functional as F
from utils import compute_returns_to_go
import timm
from torch.nn.utils.rnn import pad_sequence



class DecisionTransformerAgent:
    def __init__(self, state_size, action_size, sequence_length, device, hidden_size=768):
        self.state_size = state_size
        self.action_size = action_size
        self.sequence_length = sequence_length
        self.device = device

        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True, img_size=64, num_classes=0).to(self.device)

        for param in self.vit.parameters():
            param.requires_grad = False

        # Getting the size of the image embede layer after ViT
        self.image_embedding_size = self.vit.embed_dim

        self.decision_transformer = DecisionTransformer(
            state_dim=state_size + self.image_embedding_size,
            act_dim=action_size,
            hidden_size=hidden_size,
            max_length=sequence_length
        ).to(device)

        self.optimizer = torch.optim.Adam(self.decision_transformer.parameters())


    def learn(self, experiences):
        positions, images, actions, rewards, dones = experiences
        
        # Concatenate the experiences and later passed as an input to the decision_transformer model
        positions = torch.cat(positions, dim=1).to(self.device)
        images = torch.cat(images, dim=1).to(self.device)
        actions = torch.cat(actions, dim=1).long().to(self.device)
        
        rewards = torch.cat(rewards, dim=1).to(self.device)
        dones = torch.cat(dones, dim=1).to(self.device)

        image_embeddings = self.vit(images)
        states = torch.cat((positions, image_embeddings), dim=-1)

        # Pad the sequences, means making all the sequences of same length
        actions_padded = pad_sequence(actions, batch_first=True)
        rewards_padded = pad_sequence(rewards, batch_first=True)

        # One hot encoding of the actions
        actions_onehot = F.one_hot(actions_padded.squeeze(), num_classes=self.action_size).float()

        returns_to_go = compute_returns_to_go(rewards_padded, dones)

        timesteps = torch.arange(rewards.shape[0], dtype=torch.long).unsqueeze(0).to(self.device)

        print("States shape:", states.shape)
        print("Actions onehot shape:", actions_onehot.shape)
        print("Rewards shape:", rewards.shape)
        print("Returns to go shape:", returns_to_go.shape)
        print("Timesteps shape:", timesteps.shape)

        state_preds, action_preds, return_preds = self.decision_transformer(
            states, actions_onehot, rewards, returns_to_go, timesteps=timesteps
        )

        loss = (
            F.mse_loss(state_preds, states[:, 1:])
            + F.mse_loss(action_preds, actions[:, 1:])
            + F.mse_loss(return_preds, returns_to_go[:, 1:])
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def act(self, state, action_history, reward_history):
        position = torch.tensor(np.array(state['position']), dtype=torch.float32).unsqueeze(0).to(self.device)
        image = state['image'].unsqueeze(0).to(self.device)

        #action_history = torch.tensor(np.array(action_history), dtype=torch.float32).unsqueeze(0).to(self.device)
        action_history_tensor = torch.tensor(np.array(action_history), dtype=torch.int64).to(self.device)

        action_history_onehot = F.one_hot(action_history_tensor, num_classes=self.action_size).squeeze(0).float()

        reward_history = torch.tensor(np.array(reward_history), dtype=torch.float32).unsqueeze(0).to(self.device)

        if len(action_history) > self.sequence_length:
            #action_history = action_history[:, -self.sequence_length:]
            action_history_onehot = action_history_onehot[-self.sequence_length:]

            reward_history = reward_history[:, -self.sequence_length:]

        image_embedding = self.vit(image)

        # Concatenate the position and image embedding
        state_combined = torch.cat((position, image_embedding), dim=-1)
        returns_to_go = compute_returns_to_go(reward_history, dones=None)
        timesteps = torch.arange(len(reward_history), dtype=torch.long).unsqueeze(0).to(self.device)

        action = self.decision_transformer.get_action(
            state_combined, action_history_onehot.unsqueeze(0), reward_history.view(1, -1, 1), 
            returns_to_go=returns_to_go, timesteps=timesteps
        )

        return int(action.cpu().detach().numpy().argmax(axis=-1))

