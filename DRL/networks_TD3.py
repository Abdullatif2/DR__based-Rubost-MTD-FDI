import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from configs.config import train_config, ddpg_paths, td3_paths

# Set random seeds for reproducibility
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
T.manual_seed(seed_value)
if T.cuda.is_available():
    T.cuda.manual_seed_all(seed_value)
    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = False

if train_config['agent_type'] == 'DDPG':
    dir = 'ddpg'
    dir_actor = ddpg_paths['actor_model_dir']
    dir_critic = ddpg_paths['critic_model_dir']
elif train_config['agent_type'] == 'TD3':
    dir = 'td3'
    dir_actor = td3_paths['actor_model_dir']
    dir_critic = td3_paths['critic_model_dir']

# Actor Network with Dropout and Batch Normalization
class ActorNetwork(nn.Module):
    def __init__(self, ratio, alpha, input_dims, fc1_dims, fc2_dims, fc3_dims, fc4_dims,
                 n_actions, name, chkpt_dir=dir_actor):

        super(ActorNetwork, self).__init__()
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name + f'_{dir}')
        self.checkpoint_file = self.checkpoint_file.replace('\\', '/')
        self.ratio = ratio

        # Network architecture without batch normalization or dropout
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, fc3_dims)
        self.output = nn.Linear(fc3_dims, n_actions)

        # Optimizer with weight decay and learning rate scheduler
        self.optimizer = optim.Adam(self.parameters(), lr=alpha, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action = self.ratio * T.tanh(self.output(x))
        return action

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

# Critic Network with Dropout and Batch Normalization
class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, fc3_dims,
                 n_actions, name, chkpt_dir=dir_critic):
        super(CriticNetwork, self).__init__()
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + f'_{dir}')
        self.checkpoint_file = self.checkpoint_file.replace('\\', '/')

        # # Network architecture
        self.fc1 = nn.Linear(input_dims + n_actions, fc1_dims)
        self.bn1 = nn.LayerNorm(fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.bn2 = nn.LayerNorm(fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, fc3_dims)
        self.bn3 = nn.LayerNorm(fc3_dims)
        self.q_value = nn.Linear(fc3_dims, 1)

        # Optimizer with increased weight decay and adjusted scheduler
        self.optimizer = optim.Adam(self.parameters(), lr=beta, weight_decay=1e-3)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        x = T.cat([state, action], dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.dropout(x, p=0.2, training=self.training)
        q_value = self.q_value(x)
        return q_value

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        print(self.checkpoint_file)
        self.load_state_dict(T.load(self.checkpoint_file))

