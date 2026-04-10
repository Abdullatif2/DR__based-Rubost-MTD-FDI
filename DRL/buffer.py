import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0

        # Initialize arrays with dtype=np.float32
        self.state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)  # or dtype=bool for newer NumPy versions

    def store_transition(self, state, action, reward, state_, done):
        state_ = np.squeeze(state_)
        state = np.squeeze(state)
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state.astype(np.float32)
        self.action_memory[index] = action.astype(np.float32)
        self.reward_memory[index] = np.float32(reward)
        self.new_state_memory[index] = state_.astype(np.float32)
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones
