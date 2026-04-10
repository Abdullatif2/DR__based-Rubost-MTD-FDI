import numpy as np
import torch as T

class AWGNActionNoise():
    def __init__(self, mu, sigma_base=0.00001, sigma_ratio=0.5, sigma_min_base=0.000001, sigma_min_ratio=0.00001, decay=0.99, dfacts_index=None):
        self.initial_mu = mu
        self.mu = mu
        self.sigma_base = sigma_base
        self.sigma_ratio = sigma_ratio
        self.sigma_min_base = sigma_min_base
        self.sigma_min_ratio = sigma_min_ratio
        self.decay = decay
        self.dfacts_index = dfacts_index if dfacts_index is not None else np.array([])

        # Initialize sigma for each element of the action vector
        self.sigma = np.ones_like(mu) * sigma_base * sigma_ratio # non-dfacts index
        # self.sigma[self.dfacts_index] *= sigma_ratio # dfacts index

        # Initialize sigma_min for each element of the action vector
        self.sigma_min = np.ones_like(mu) * sigma_min_base * sigma_min_ratio
        # self.sigma_min[self.dfacts_index] *= sigma_min_ratio # dfacts index

    def __call__(self):
        # Generate noise with different sigma values based on index
        noise = self.mu + self.sigma * np.random.normal(size=self.mu.shape)

        # Decay the sigma values
        self.sigma = np.maximum(self.sigma_min, self.sigma * self.decay)
        
        return noise

   
    def reset(self):
        # Reset sigma values to initial values
        self.sigma = np.ones_like(self.mu) * self.sigma_base * self.sigma_ratio
        # self.sigma[self.dfacts_index] *= self.sigma_ratio
        # Reset the decay-adjusted minimum sigma values
        self.sigma_min = np.ones_like(self.mu) * self.sigma_min_base * self.sigma_min_ratio
        # self.sigma_min[self.dfacts_index] *= self.sigma_min_ratio









#%%
# import numpy as np
# import torch as T
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F

# class AWGNActionNoise():
#     def __init__(self, mu, sigma=0.00001, sigma_min=0.000001, decay=0.99):
#         self.initial_mu = mu
#         self.initial_sigma = sigma
#         self.mu = mu
#         self.sigma = sigma
#         self.sigma_min = sigma_min
#         self.decay = decay

#     def __call__(self):
#         noise = self.mu + self.sigma * np.random.normal(size=self.mu.shape)
#         self.sigma = max(self.sigma_min, self.sigma * self.decay)
#         return noise

#     def reset(self):
#         # Reset noise parameters to initial values
#         self.mu = self.initial_mu
#         self.sigma = self.initial_sigma



