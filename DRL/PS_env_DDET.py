import math
import random
import numpy as np
from copy import deepcopy
from numpy.linalg import norm
import torch as T
import sys
sys.path.append('./')

from configs.config import train_config

# Set random seeds for reproducibility
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
T.manual_seed(seed_value)
if T.cuda.is_available():
    T.cuda.manual_seed_all(seed_value)
    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = False

class CustomEnv:
    def __init__(self, case_class, max_steps, ratio, dfacts_index, observation_space, mode):
        # Case (IEEE-bus system) related
        self.case_class = deepcopy(case_class)
        self.mode_ = mode
        self.dfacts_index = dfacts_index
        self.step_count = 0
        self.max_steps = max_steps
        self.x = deepcopy(self.case_class.x)
        self.observation_space = np.zeros(observation_space)
        self.obs = np.zeros(observation_space)

        # Set max and min constraints for all branches (noise)
        self.x_max = self.x * 1.0001
        self.x_min = self.x * 0.9999

        # Perturbation (20%) applied for branches with D-FACTS
        self.ratio = ratio
        self.x_max[self.dfacts_index] *= (1 + ratio)
        self.x_min[self.dfacts_index] *= (1 - ratio)

        self.done = False
        self.reward = 0

        # Inputs
        self.z_noise = []
        self.v_est = []
        self.load_active = []
        self.load_reactive = []

        # To save results when testing
        self.x_ratio_ = {}
        self.mtd_eff_ = {}
        self.mtd_hidden_ = {}
        self.cost_no_mtd_ = {}
        self.cost_with_mtd_ = {}
        self.residual_normal_ = {}
        self.residual_BDD_ = {}

    def step(self, action, state):
        self.z_noise = state[0]
        self.v_est = state[1]
        self.load_active = state[2]
        self.load_reactive = state[3]

        self.step_count += 1

        # Apply action to x_mtd
        x_mtd = deepcopy(self.x)
        x_mtd[self.dfacts_index] = x_mtd[self.dfacts_index] * (1 + action)
        assert np.all(x_mtd <= self.x_max + 1e-6)
        assert np.all(x_mtd >= self.x_min - 1e-6)

        if self.mode_ == 'test':
            z_noise_new, v_est_ope_no_att, result_new, result, x_ratio, mtd_hidden, cost_no_mtd, cost_with_mtd, residual_normal, mtd_eff, residual_BDD = \
                mtd_metric_with_attack(self.case_class, x_mtd, self.v_est, self.load_active, self.load_reactive, self.ratio, self.dfacts_index)
            self.aggregate_results(x_ratio, mtd_hidden, cost_no_mtd, cost_with_mtd, residual_normal, mtd_eff, residual_BDD)
            
            
            self.reward = self.calculate_reward_base_theta(x_mtd)
            self.obs = z_noise_new 
        else:
            # Training mode
            z_noise_new = mtd_training(self.case_class, x_mtd, self.v_est,
                                       self.load_active, self.load_reactive)
            self.reward = self.calculate_reward_base_theta(x_mtd)
            self.obs = z_noise_new

        if self.step_count >= self.max_steps:
            self.done = True
        self.obs = self.obs.astype(np.float32)

        return self.obs, self.reward, self.done

    def x_to_b(self, case, x):
        return -x / (case.r ** 2 + x ** 2)

    def find_singular_val(self, x_mtd):
        # Before MTD matrix
        v_est = self.v_est.flatten()
        H_pre = self.case_class.H_v_hat(v_est)
        H_pre = self.case_class.R_inv_12_ @ H_pre  # Normalize
        P_pre, S_pre = self.case_class.mtd_matrix(H_pre)  # Projection matrices

        # Post-MTD matrix
        b_mtd = self.x_to_b(self.case_class, x_mtd)
        V, C, Ars, Arc = self.case_class.H_v_hat_robust(self.v_est.numpy())
        C = self.case_class.R_inv_12_ @ C
        V = self.case_class.R_inv_12_ @ V
        A = Arc
        H_post = C + V @ np.diag(b_mtd) @ A
        P_post = H_post @ np.linalg.inv(H_post.T @ H_post) @ H_post.T

        composite_matrix = np.concatenate([H_pre, H_post], axis=-1)
        rank_com = np.linalg.matrix_rank(composite_matrix)
        k_min = 2 * (self.case_class.no_bus - 1) - rank_com

        # Perform SVD
        U, singular_values, V_transpose = np.linalg.svd(P_pre @ P_post)
        threshold = 1e-10

        # The largest non-one singular value
        singular_value_non_one = singular_values[k_min]

        return P_post, P_pre, H_post, H_pre, C, V, A, singular_value_non_one, singular_values

    def calculate_reward_base_theta(self, x_mtd):
        P_post, P_pre, H_post, H_pre, C, V, A, singular_value_non_one, singular_values = self.find_singular_val(x_mtd)
        k_min = 2 * (self.case_class.no_bus - 1) - np.linalg.matrix_rank(np.concatenate([H_post, H_pre], axis=-1))

        # Calculate angles in degrees
        theta_deg = np.degrees(np.arccos(np.clip(np.sqrt(singular_values[k_min:]), -1.0, 1.0)))
        # print('theta_deg',theta_deg)

        # Compute the reward
        max_reward = 1.0
        reward_scale = 1000
        diff = np.abs(theta_deg - 90)
        reward = (max_reward - diff / 90) * reward_scale
        final_reward = np.sum(reward * 100)

        # Normalize reward
        final_reward /= 10000
        return final_reward

    def aggregate_results(self, x_ratio, mtd_hidden, cost_no_mtd, cost_with_mtd, residual_normal, mtd_eff, residual_BDD):
            for key, value in x_ratio.items():
                if key not in self.x_ratio_:
                    self.x_ratio_[key] = []
                self.x_ratio_[key].extend(value)
            for key, value in mtd_hidden.items():
                if key not in self.mtd_hidden_:
                    self.mtd_hidden_[key] = []
                self.mtd_hidden_[key].extend(value)
            for key, value in cost_no_mtd.items():
                if key not in self.cost_no_mtd_:
                    self.cost_no_mtd_[key] = []
                self.cost_no_mtd_[key].extend(value)
            for key, value in cost_with_mtd.items():
                if key not in self.cost_with_mtd_:
                    self.cost_with_mtd_[key] = []
                self.cost_with_mtd_[key].extend(value)
            for key, value in residual_normal.items():
                if key not in self.residual_normal_:
                    self.residual_normal_[key] = []
                self.residual_normal_[key].extend(value)
            for key, value in mtd_eff.items():
                if key not in self.mtd_eff_:
                    self.mtd_eff_[key] = []
                self.mtd_eff_[key].extend(value)
            for key, value in residual_BDD.items():
                if key not in self.residual_BDD_:
                    self.residual_BDD_[key] = []
                self.residual_BDD_[key].extend(value)
