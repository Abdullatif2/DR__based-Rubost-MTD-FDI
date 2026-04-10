import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import random
from DRL.noise import AWGNActionNoise
from DRL.networks_TD3 import ActorNetwork, CriticNetwork
from DRL.buffer import ReplayBuffer
from configs.config import train_config
from torch.profiler import profile
import time
# Set random seeds for reproducibility
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
T.manual_seed(seed_value)
if T.cuda.is_available():
    T.cuda.manual_seed_all(seed_value)
    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = False

class Agent:
    def __init__(self, ratio, alpha, beta, input_dims, tau, n_actions,
                 Act_fc1_dims, Act_fc2_dims, Act_fc3_dims, Act_fc4_dims,
                 Crtc_fc1_dims, Crtc_fc2_dims, Crtc_fc3_dims, Crtc_fc4_dims,
                 batch_size=128, gamma=0.99, max_size=1e6):
        self.gamma = gamma
        self.tau = tau  # Reduced tau for smoother updates
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.max_size = int(max_size)
        self.alpha = alpha
        self.beta = beta  # Reduced beta for critic's learning rate
        self.time_step = 0
        self.warmup_steps = 100
        self.min_action = -ratio
        self.max_action = ratio

        # Noise parameters for exploration
        # self.initial_noise_scale = 0.1
        # self.final_noise_scale = 0.01
        # self.noise_decay = (self.initial_noise_scale - self.final_noise_scale) / 100000

        self.initial_noise_scale = 0.05
        self.final_noise_scale = 0.001
        self.noise_decay = (self.initial_noise_scale - self.final_noise_scale) / 500


        self.memory = ReplayBuffer(self.max_size, input_dims, n_actions)

        # Actor and Critic Networks with modified learning rates and regularization
        self.actor = ActorNetwork(ratio, self.alpha, input_dims, Act_fc1_dims, Act_fc2_dims,
                                  Act_fc3_dims, Act_fc4_dims, n_actions=n_actions, name='actor')
        self.critic = CriticNetwork(self.beta, input_dims, Crtc_fc1_dims, Crtc_fc2_dims,
                                    Crtc_fc3_dims, n_actions=n_actions, name='critic')

        self.target_actor = ActorNetwork(ratio, self.alpha, input_dims, Act_fc1_dims, Act_fc2_dims,
                                         Act_fc3_dims, Act_fc4_dims, n_actions=n_actions, name='target_actor')
        self.target_critic = CriticNetwork(self.beta, input_dims, Crtc_fc1_dims, Crtc_fc2_dims,
                                           Crtc_fc3_dims, n_actions=n_actions, name='target_critic')

        # Copy weights from original networks to target networks
        self.update_network_parameters(tau=1.0)

    def choose_action(self, state, episode, is_training=True):
        # Ensure state is a PyTorch tensor and of type float32
        if not isinstance(state, T.Tensor):
            state = T.tensor(state, dtype=T.float32).to(self.actor.device)
        else:
            state = state.to(self.actor.device)
            state = state.float()  # Convert to float32 if it's not already

        # Add batch dimension if missing
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Shape: [1, input_dims]

        self.actor.eval()
        start_time = time.time()  ## record the time
        with T.no_grad():
            # with profile() as prof:  # Start profiling   
            mu = self.actor.forward(state)
            # print(prof.key_averages().table(sort_by="cpu_time_total"))  # Sort by CPU time
        end_time = time.time()
        

        # Remove batch dimension
        mu = mu.squeeze(0)

        if is_training & episode<50: # removing noise after 40 ep
            noise_scale = max(self.final_noise_scale, self.initial_noise_scale - self.noise_decay * self.time_step)
            noise = T.normal(0, noise_scale, size=mu.shape, dtype=T.float32).to(self.actor.device)
            mu += noise
            mu = T.clamp(mu, self.min_action, self.max_action)
            self.time_step += 1

        return mu.cpu().numpy(), (end_time - start_time)

    def remember(self, state, action, reward, state_, done):
        # Convert tensors to NumPy arrays
        if isinstance(state, T.Tensor):
            state = state.detach().cpu().numpy()
        if isinstance(action, T.Tensor):
            action = action.detach().cpu().numpy()
        if isinstance(reward, T.Tensor):
            reward = reward.item()
        if isinstance(state_, T.Tensor):
            state_ = state_.detach().cpu().numpy()
        if isinstance(done, T.Tensor):
            done = done.item()

        # Ensure state is a 1D array
        state = np.squeeze(state)
        state_ = np.squeeze(state_)

        self.memory.store_transition(state, action, reward, state_, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return None,None

        states, actions, rewards, states_, done = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float32).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float32).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float32).to(self.actor.device).unsqueeze(1)
        states_ = T.tensor(states_, dtype=T.float32).to(self.actor.device)
        done = T.tensor(done, dtype=T.bool).to(self.actor.device)

        # Normalize rewards (optional but recommended)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Update Critic Network
        with T.no_grad():
            target_actions = self.target_actor.forward(states_)
            target_critic_value = self.target_critic.forward(states_, target_actions)
            target_critic_value[done] = 0.0
            target = rewards + self.gamma * target_critic_value

        self.critic.optimizer.zero_grad()
        critic_value = self.critic.forward(states, actions)
        # critic_loss = F.mse_loss(critic_value, target)
        critic_loss = F.l1_loss(critic_value, target)

        
        critic_loss.backward()
        T.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.1)  # Clip gradients
        self.critic.optimizer.step()

        # Update Actor Network
        self.actor.optimizer.zero_grad()
        actor_actions = self.actor.forward(states)
        actor_loss = -self.critic.forward(states, actor_actions).mean()  # Actor loss should be negative
        actor_loss.backward()

        # Check gradient flow in actor network
        # for param in self.actor.parameters():
        #     if param.grad is None:
        #         print(f"No gradient flow for {param.name}")
        #     else:
        #         print(f"Gradient for {param.name}: {param.grad.norm()}")

        T.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.1)  # Clip gradients
        self.actor.optimizer.step()

        # Soft update target networks
        self.update_network_parameters()

        # Compare actions from actor and target actor
        with T.no_grad():
            target_actor_actions = self.target_actor.forward(states)
        action_difference = (actor_actions - target_actor_actions).abs().mean().item()

        # Optional: Print losses and action difference for monitoring
        # print(f"Actor Loss: {actor_loss.item()}, Critic Loss: {critic_loss.item()}, Action Difference: {action_difference}")
        return actor_loss.item(),critic_loss.item()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        # Soft update for actor network
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        # Soft update for critic network
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
