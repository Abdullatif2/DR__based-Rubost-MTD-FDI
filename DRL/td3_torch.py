# from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch as T
import torch.nn.functional as F
import numpy as np
import torch as T
import random
import torch.nn.functional as F
from DRL.noise import AWGNActionNoise
from DRL.networks_TD3 import ActorNetwork, CriticNetwork
from DRL.buffer import ReplayBuffer
from configs.config import train_config
from torch.profiler import profile
import time
T.set_printoptions(precision=8)

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
T.manual_seed(seed_value)
if T.cuda.is_available():
    T.cuda.manual_seed_all(seed_value)
    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = False
from torch.profiler import profile 


class Agent:

    def __init__(self, ratio, alpha, beta, input_dims, tau, n_actions,
                Act_fc1_dims, Act_fc2_dims, Act_fc3_dims, Act_fc4_dims,
                Crtc_fc1_dims, Crtc_fc2_dims, Crtc_fc3_dims, Crtc_fc4_dims,
                batch_size=128, gamma=0.99, max_size=1e6):
    
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.max_size = int(max_size)
        self.alpha = alpha 
        self.beta = beta
        self.learn_step_cntr = 0
        self.time_step = 0
        # self.warmup_steps = 100
        self.update_actor_iter = 2
        self.min_action  = -ratio
        self.max_action  = ratio

        # Noise parameters for exploration
        self.initial_noise_scale = 0.1
        self.final_noise_scale = 0.01
        self.noise_decay = (self.initial_noise_scale - self.final_noise_scale) / 100000

        # self.initial_noise_scale = 0.05
        # self.final_noise_scale = 0.001
        # self.noise_decay = (self.initial_noise_scale - self.final_noise_scale) / 500
        


        self.memory = ReplayBuffer(self.max_size, input_dims, n_actions)
        # self.noise = AWGNActionNoise(mu=np.zeros(n_actions), sigma_base=0.00001, sigma_ratio=ratio, dfacts_index=dfacts_index)

   # Actor and Critic Networks with modified learning rates and regularization
        self.actor = ActorNetwork(ratio, self.alpha, input_dims, Act_fc1_dims, Act_fc2_dims, Act_fc3_dims, Act_fc4_dims, n_actions=n_actions, name='actor')
        self.critic_1 = CriticNetwork(self.beta, input_dims, Crtc_fc1_dims, Crtc_fc2_dims, Crtc_fc3_dims, n_actions=n_actions, name='critic_1')
        self.critic_2 = CriticNetwork(self.beta, input_dims, Crtc_fc1_dims, Crtc_fc2_dims, Crtc_fc3_dims, n_actions=n_actions, name='critic_2')

        self.target_actor = ActorNetwork(ratio, self.alpha, input_dims, Act_fc1_dims, Act_fc2_dims, Act_fc3_dims, Act_fc4_dims, n_actions=n_actions, name='target_actor')
        self.target_critic_1 = CriticNetwork(self.beta, input_dims, Crtc_fc1_dims, Crtc_fc2_dims, Crtc_fc3_dims, n_actions=n_actions, name='target_critic_1')
        self.target_critic_2 = CriticNetwork(self.beta, input_dims, Crtc_fc1_dims, Crtc_fc2_dims, Crtc_fc3_dims, n_actions=n_actions, name='target_critic_2')

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
        end_time = time.time()
        # Remove batch dimension
        mu = mu.squeeze(0)
        self.actor.train()
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
            return None, None  #-------
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        # Convert to PyTorch tensors and move to the correct device
        state = T.tensor(state, dtype=T.float32).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float32).to(self.critic_1.device)
        reward = T.tensor(reward, dtype=T.float32).to(self.critic_1.device).unsqueeze(1)  #-------
        state_ = T.tensor(new_state, dtype=T.float32).to(self.critic_1.device)
        done = T.tensor(done, dtype=T.bool).to(self.critic_1.device)  #-------

        # Normalize rewards (optional but recommended)
        reward = (reward - reward.mean()) / (reward.std() + 1e-8)  #-------

        # Compute target actions with noise and clamp
        with T.no_grad():
            target_actions = self.target_actor.forward(state_)
            target_actions = target_actions + T.clamp(T.tensor(np.random.normal(scale=0.05)), -0.05, 0.05)
            target_actions = T.clamp(target_actions, self.min_action, self.max_action)

            q1_ = self.target_critic_1.forward(state_, target_actions)
            q2_ = self.target_critic_2.forward(state_, target_actions)

            q1_[done] = 0.0
            q2_[done] = 0.0

            q1_ = q1_.view(-1)
            q2_ = q2_.view(-1)

            critic_value_ = T.min(q1_, q2_).unsqueeze(1)
            target = reward + self.gamma * critic_value_

            
        # Update Critic Networks
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)

        # q1_loss = F.mse_loss(q1, target)
        # q2_loss = F.mse_loss(q2, target)

        # # L1 loss
        q1_loss = F.l1_loss(q1, target)
        q2_loss = F.l1_loss(q2, target)


        # # Huber Loss
        # q1_loss = F.smooth_l1_loss(q1, target)
        # q2_loss = F.smooth_l1_loss(q2, target)
        
        
        
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()

        # Clip gradients for stability
        T.nn.utils.clip_grad_norm_(self.critic_1.parameters(), max_norm=0.1)  #-------
        T.nn.utils.clip_grad_norm_(self.critic_2.parameters(), max_norm=0.1)  #-------
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr += 1

        # Delayed policy update
        if self.learn_step_cntr % self.update_actor_iter != 0:
            return None, critic_loss.item()  # Return critic loss only if actor is not updated

        # Update Actor Network
        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        actor_loss = -T.mean(actor_q1_loss)  
        actor_loss.backward()

        # Clip gradients for actor
        T.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.1)  #-------
        self.actor.optimizer.step()

        # Soft update target networks
        self.update_network_parameters()

        # Optional: Return losses for monitoring
        return actor_loss.item(), critic_loss.item()  # Return both actor and critic loss


    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        # Soft update for actor network
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        # Soft update for critic_1 network
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        # Soft update for critic_2 network
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        
    
    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()


