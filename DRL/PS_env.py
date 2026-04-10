# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 10:27:43 2023

@author: comla
the version is purely for training (no dataset split into training and testing)
"""

from copy import deepcopy
from numpy.linalg import norm
import torch as T
import numpy as np
import sys
sys.path.insert(0, 'C:/Users/comla/OneDrive/Desktop/Robust_MTD-main_Copy')
from utils.evaluation_fun import generate_data
from utils.initialization import *
from utils.utils import x_to_b, find_posi


import numpy as np
from numpy.linalg import LinAlgError

def _orth_basis(A, tol=None):
    """
    Return an orthonormal basis Q_r for Col(A) via thin QR.
    Works even if A is rank-deficient. Q_r has shape (m, r).
    """
    A = np.asarray(A, dtype=np.float64)
    Q, R = np.linalg.qr(A, mode='reduced')              # J = Q R
    if R.size == 0:
        return Q[:, :0]
    if tol is None:
        eps = np.finfo(R.dtype).eps
        smax = np.abs(np.diag(R)).max() if R.ndim == 2 and R.shape[0] and R.shape[1] else 0.0
        tol = eps * max(A.shape) * (smax if smax > 0 else 1.0)
    r = int(np.sum(np.abs(np.diag(R)) > tol))            # numerical rank
    return Q[:, :r] if r > 0 else Q[:, :0]

def _principal_angles_from_jacobians(J1, J2, svd_eps=1e-12):
    """
    Principal angles between Col(J1) and Col(J2).
    Returns (theta, s) where s = cos(theta), theta in radians.
    """
    Q1 = _orth_basis(J1)
    Q2 = _orth_basis(J2)
    if Q1.size == 0 or Q2.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    C = Q1.T @ Q2                                        # small (r1 x r2)
    try:
        s = np.linalg.svd(C, compute_uv=False)           # s in [0, 1]
    except LinAlgError:
        s = np.linalg.svd(C + svd_eps*np.eye(*C.shape), compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    theta = np.arccos(s)
    return theta, s





# CustomEnv_robust
class CustomEnv:
    def __init__(
            self,
            case,
            max_steps,
            ratio, dfacts_index, observation_space, mode):
        # # case (IEEE-bus sytem) related
        self.mode = mode 
        self.case = case
        self.observation_space = np.zeros(observation_space)  # Adjust based on the actual structure
        # self.observation_space = np.zeros(116)  # Adjust based on the actual structure

        self.mode_ = 'train'
        self.dfacts_index = dfacts_index #np.array([2, 3, 4, 12, 15, 18, 20]) - 1
        self.action_space = np.zeros(len(self.dfacts_index))  # assuming 7 D_facts
        self.step_count = 0
        self.max_steps = max_steps
        self.x = self.case.x
        self.x_max = self.x * 1.0001 # set max and min constraints for all branches (noise)
        self.x_min = self.x * 0.9999
        self.ratio = ratio # perturbation applied for branches with D-FACTS
        self.x_max[self.dfacts_index] *= (1 + ratio)        
        self.x_min[self.dfacts_index] *= (1 - ratio)
        self.done = False
        self.theta_deg = 0
        self.z_noise_new = np.zeros((68, 1))
        self.v_est = np.zeros(14)
        self.reward = 0        
        self.COST_pre = []
        self.COST_after = []
        self.cost_pre_mtd = []
        self.cost_post_mtd = []
        self.x_ratio = []
        self.residual_SO_list = []
        self.residual_attacker_list = []
        self.z_noise_o = []
        self.v_est = []
        self.load_active = []
        self.load_reactive = []
        self.pv_active_ = []
        self.pv_reactive_ = []
        self.batch_size = 400
        self.idx = 0
        self.batch_index = 0
        self.attack_status = 0
        

       # Taraining with Dataset (without attack)
 
        self.result_new_success = []
        self.x_mtd_change_ratio = []
        self.residual_ope_no_at = []
        self.result_f = []
        self.result_new_f = []

        # Testing with Dataset (with attack)
        self.x_ratio = {}
        self.mtd_eff = {}        # The residual of operator at stage one
        self.mtd_hidden = {}
        self.cost_no_mtd = {}
        self.cost_with_mtd = {}
        self.residual_normal = {}
        self.residual_BDD = {}
        self.residual_MR = {}
        self.MAG_CHANGE = []
        self.ANG_CHANGE = []
        # for evaluation using the R-MTD
        self.RESIDUAL_ATTACKER_DRL = [[],[],[],[],[],[]]
        self.RESIDUAL_DRL = [[],[],[],[],[],[]]
        self.ATTACK_POSI_DRL = np.zeros(len(attack_strength_under_test)+1,)
        self.TP_RANDOM_DRL = np.zeros(len(attack_strength_under_test)+1,)


    def reset(self):
        self.idx = 0
        self.step_count = 0
        self.done = False
        
        self.x_max = self.x * 1.0001
        self.x_min = self.x * 0.9999
        self.x_max[self.dfacts_index] *= (1 + self.ratio)
        self.x_min[self.dfacts_index] *= (1 - self.ratio)

        active_power, reactive_power, v_mag_est, v_ang_est, Jr_N, result = generate_data(case)
        result, z_new, z_noise = self.case.ac_opf(active_power, reactive_power)
        obs = z_noise
        return obs



    def step(self, action, state):

        def _safe_ratio(num, den, eps=1e-12):
            num = np.asarray(num, dtype=np.float64)
            den = np.asarray(den, dtype=np.float64)
            out = np.divide(num, den, out=np.zeros_like(num, dtype=np.float64),
                            where=np.isfinite(den) & (np.abs(den) > eps))
            out[~np.isfinite(out)] = 0.0
            return out


        active_power = state[0].cpu().numpy()
        reactive_power = state[1].cpu().numpy()
        v_mag_est = state[2].cpu().numpy()
        v_ang_est = state[3].cpu().numpy()

        # print("action", action)
        x_mtd = deepcopy(self.x)
        x_mtd[self.dfacts_index] = x_mtd[self.dfacts_index] * (1 + action)
        assert np.all(x_mtd <= self.x_max + 1e-6)
        assert np.all(x_mtd >= self.x_min - 1e-6)
        
        # ------------------------------------- Build post-MTD grid -------------------------------------

        mpc_new = deepcopy(mpc)
        mpc_new['branch'][:,3] = x_mtd
        case_new = dc_grid(mpc_new, gencost, genlimit_max, genlimit_min, flowlimit)
        case_new.prepare_BDD_variable(alpha, R, R_)

        # ------------------------------------- Reward (stable angles) ----------------------------------
        reward = self.calculate_reward_base_theta(v_mag_est, v_ang_est, x_mtd)
        # ------------------------------------- AC-OPF roll-out -----------------------------------------

        result_new, z_new, self.z_noise_new = case_new.ac_opf(active_power, reactive_power)
        v_mag_est_new, v_ang_est_new = result_new['bus'][:,7], result_new['bus'][:,8]*np.pi/180 # find the state
        self.COST_after.append(result_new['f'])
        # self.MAG_CHANGE.append((v_mag_est_new - v_mag_est)/v_mag_est_new)
        # self.ANG_CHANGE.append((v_ang_est_new - v_ang_est)/v_ang_est_new)
        self.MAG_CHANGE.append(_safe_ratio(v_mag_est_new - v_mag_est, v_mag_est_new))
        self.ANG_CHANGE.append(_safe_ratio(v_ang_est_new - v_ang_est, v_ang_est_new))

        # hiddenness (evaluate residual on original reactance)
        residual_attacker, z_est_attacker = self.case.find_lambda_residual_ac(self.case.x, self.z_noise_new, result_new)     # on the original reactance

        # test_no = 10
        # attack and detection
        if self.mode == "test":
            self.RESIDUAL_ATTACKER_DRL.append(residual_attacker)
            # generate attack using post-mtd measurement and ORIGINAL reactance
            # for j in range(test_no):
            #     # print(" i am insode the loop and it is: ", j)
            #     # generate attack using post-mtd measurement and ORIGINAL reactance
            #     z_noise_a = self.case.random_ac_attack(v_mag_est_new, v_ang_est_new, self.z_noise_new, att_state_ratio)  # using post-mtd measurement and ORIGINAL reactance
            #     residual, z_est_a = case_new.find_lambda_residual_ac(x_mtd, z_noise_a, result_new)                
            #     attack_strength = norm(z_noise_a - self.z_noise_new, 2)/np.sqrt(np.sum(case.R))
            #     posi = find_posi(attack_strength)
            #     self.RESIDUAL_DRL[posi].append(residual)
            #     self.ATTACK_POSI_DRL[posi] += 1
            #     if residual >= case_new.BDD_threshold:
            #         self.TP_RANDOM_DRL[posi] += 1

            for j in range(test_no):
                print(f"attack test: {j}/{test_no}")
                # Guarantee: Only increment j when a successful attack-residual calculation occurs
                max_attack_attempts = 10  # Optional: max tries to avoid infinite loop
                attack_ok = False
                for _ in range(max_attack_attempts):
                    try:
                        z_noise_a = self.case.random_ac_attack(v_mag_est_new, v_ang_est_new, self.z_noise_new, att_state_ratio)
                        residual, z_est_a = case_new.find_lambda_residual_ac(x_mtd, z_noise_a, result_new)
                        if np.isnan(residual) or np.isinf(residual):
                            continue
                        # Only get here if no exception and residual is valid!
                        attack_ok = True
                        break
                    except Exception as e:
                        # Log if you want: print(f"Attack {j} failed: {e}")
                        continue
                if not attack_ok:
                    print(f"[Warning] Skipped attack {j} after {max_attack_attempts} failures.")
                    continue
                # -- rest of your code as before --
                attack_strength = norm(z_noise_a - self.z_noise_new, 2)/np.sqrt(np.sum(case.R))
                posi = find_posi(attack_strength)
                self.RESIDUAL_DRL[posi].append(residual)
                self.ATTACK_POSI_DRL[posi] += 1
                if residual >= case_new.BDD_threshold:
                    self.TP_RANDOM_DRL[posi] += 1


        
        self.step_count += 1
        if (self.step_count >= self.max_steps):
            self.done = True
    
        # self.reward = reward
        print("reward = ", reward)
        return self.z_noise_new, reward, self.done


    def sing_val_robust(self, v_mag, v_ang, x_mtd):
        """
        Build J before/after MTD, compute principal angles from orthonormal bases.
        Also return light diagnostics (projector-like norms via Q, no inverses).
        """
        # ---- ensure float64 for stable LA
        v_mag = np.asarray(v_mag, dtype=np.float64)
        v_ang = np.asarray(v_ang, dtype=np.float64)
        x_mtd = np.asarray(x_mtd, dtype=np.float64)

        # ---- Jacobians
        _, _, _, Jr_N = self.case.jacobian(v_mag, v_ang, self.case.b)
        b_mtd = x_to_b(self.case, x_mtd)
        _, _, _, Jr_mtd_N = self.case.jacobian(v_mag, v_ang, b_mtd)

        # ---- principal angles (theta, s=cos(theta))
        theta, s = _principal_angles_from_jacobians(Jr_N, Jr_mtd_N)

        # ---- diagnostics using orthonormal projectors P̃ = Q Q^T (no inv)
        Q1 = _orth_basis(Jr_N)
        Q2 = _orth_basis(Jr_mtd_N)
        if Q1.size and Q2.size:
            P1_tilde = Q1 @ Q1.T
            P_tilde  = Q2 @ Q2.T
            l2_norm  = np.linalg.norm(P1_tilde @ P_tilde, 2)
            fro_norm = np.linalg.norm(P1_tilde @ P_tilde, 'fro')
        else:
            l2_norm, fro_norm = 0.0, 0.0

        # ---- “largest non-one” singular value (optional metric)
        eps_shared = 1.0 - 1e-8
        k_min = int(np.sum(s >= eps_shared))
        sv_non_one = s[k_min] if k_min < len(s) else 0.0

        # keep return signature rich for downstream usage/logging
        return sv_non_one, l2_norm, fro_norm, theta, s


    def calculate_reward_base_theta(self, v_mag, v_ang, x_mtd):
        """
        Reward pushes principal angles toward 90° on the non-shared subspace.
        - auto-detects k_min by thresholding s ≈ 1
        - dimension-agnostic (works for 14, 57, ...)
        - keeps magnitude in a similar ballpark to your old scaling
        """
    # try:
        _, l2_norm, fro_norm, theta, s = self.sing_val_robust(v_mag, v_ang, x_mtd)

        if theta.size == 0:
            return -0.1  # degenerate (no identifiable subspace)

        # ---- identify current intersection size (shared directions ~ 1)
        eps_shared = 1.0 - 1e-8               # tighten to 1-1e-10 if needed
        k_min = int(np.sum(s >= eps_shared))  # adaptive, no hardcoding

        # ---- work only on separable part
        theta_sep = theta[k_min:] if k_min < theta.size else np.array([], dtype=np.float64)
        if theta_sep.size == 0:
            return -0.05  # everything shared → small penalty

        # ---- per-angle score: 1 at 90°, 0 at 0°
        ri = 1.0 - np.abs(theta_sep - (0.5*np.pi)) / (0.5*np.pi)
        ri = np.clip(ri, 0.0, 1.0)

        # emphasize the hardest angles a bit (like your focus on a subset)
        ri_sorted = np.sort(ri)                       # ascending
        k_focus = min(15, len(ri_sorted))
        core = ri_sorted[:k_focus].mean()
        tail = ri.mean()

        # light penalties to discourage degeneracy (kept tiny)
        pen_l2  = np.tanh(l2_norm) / 5.0
        pen_fro = np.tanh(fro_norm / 10.0) / 5.0

        # ---- final reward (scale ~ O(1..3); your agent adapts anyway)
        reward = 2.0 * core + 1.0 * tail - pen_l2 - pen_fro
        return float(reward)

        # except Exception:
        #     # never kill the episode due to numerics
        #     return -0.1


