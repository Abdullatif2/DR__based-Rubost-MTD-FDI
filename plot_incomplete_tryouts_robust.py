
# # -*- coding: utf-8 -*-
# # %%
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn
# from scipy.stats import ncx2, chi2
# from configs.config import train_config, paths_config
# # from utils.initialization import *



# # % SETTINGS

# grid_choice = 1         ## 0:case6, 1:case14, 2:case57
# ratio_choice = 0        ## the level of D-FACTS perturbation ratio ---- 0 : pm 0.2 --- 1 : pm 0.3 --- 2 : pm 0.4 --- 3 : pm 0.5
# dfacts_choise = 0       ## the d-facts perturbation ----- 0 : perturb all lines --- 1 : minimum full rank --- 2 : minimum 
# col_choice = 0          ## the on/off of the column angle constraint ---- 0 : no column constraint 1 : with column constraint

# if ratio_choice == 0:
#     ratio = 0.2
#     x_max_ratio = 0.2 ## |delta x| <= x_max_ratio
# elif ratio_choice == 1:
#     ratio = 0.3
#     x_max_ratio = 0.3 ## |delta x| <= x_max_ratio
# elif ratio_choice == 2:
#     ratio = 0.4
#     x_max_ratio = 0.4 ## |delta x| <= x_max_ratio
# elif ratio_choice == 3:
#     ratio = 0.5 # the maximum in liturature
#     x_max_ratio = 0.5 ## |delta x| <= x_max_ratio

# elif ratio_choice == 10:
#     # only for a test
#     ratio = 0.1
#     x_max_ratio = 0.1  ## |delta x| <= x_max_ratio
#     x_min_ratio = 0.01 ## |delta_x|

# placement = ''
# if dfacts_choise == 0:
#     placement = 'FP'
# else:
#     placement = 'FR'


# ddpg_train_type = 'with_ep_schdlr_ptienc_2_0.001_0.001_dropout_03_first_2_layers'
# td3_train_type = 'with_ep_schdlr_ptienc_2_0.001_0.001_dropout_03_first_2_layers_old_noise'

# agent_type = train_config['agent_type']
# test_ratio = x_max_ratio


# name = 'case14'

# suffix = f'choice_{dfacts_choise}_ratio_{ratio_choice}_column_{col_choice}'

# if name == 'case14':
#     BDD_threshold = 58.124
# elif name == 'case57':
#     BDD_threshold = 192.700
    
# rc = {"font.family" : "serif", 
#       "mathtext.fontset" : "stix",
#       }
# plt.rcParams.update(rc)
# plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
# font = {'size'   : 25}
# plt.rc('font', **font)

# attack_strength_under_test = [5,7,10,15,20]

# x_axis_name = [str(i) for i in attack_strength_under_test]



# # % LOAD DATA


# ###########################################     AC      ##########################################################

# # RANDOM_ROBUST_AC = np.load(f'final_simulation_data/{name}/ac/Robust/{suffix}_TP_RANDOM_ROBUST.npy', allow_pickle = True)
# # RANDOM_RANDOM_AC = np.load(f'final_simulation_data/{name}/ac/Robust/{suffix}_TP_RANDOM_RANDOM.npy', allow_pickle = True)
# RESIDUAL = np.load(f'final_simulation_data/{name}/ac/Robust/{suffix}_RESIDUAL_ATTACKER_ROBUST.npy', allow_pickle = True)

# RANDOM_ddpg_AC = np.load(f'final_simulation_data/{name}/ac/DDPG_ratio_{test_ratio}/{ddpg_train_type}/{suffix}_TP_RANDOM_DRL.npy', allow_pickle = True) #----------------------
# RESIDUAL_att_ddpg = np.load(f'final_simulation_data/{name}/ac/DDPG_ratio_{test_ratio}/{ddpg_train_type}/{suffix}_RESIDUAL_ATTACKER_DRL.npy', allow_pickle = True) #----------------------

# # RANDOM_td3_AC = np.load(f'final_simulation_data/{name}/ac/TD3_ratio_{test_ratio}/{td3_train_type}/{suffix}_TP_RANDOM_DRL.npy', allow_pickle = True) #----------------------
# RESIDUAL_att_td3 = np.load(f'final_simulation_data/{name}/ac/TD3_ratio_{test_ratio}/{td3_train_type}/{suffix}_RESIDUAL_ATTACKER_DRL.npy', allow_pickle = True) #----------------------



# RESIDUAL_ROBUST = np.load(f'final_simulation_data/{name}/ac/Robust//{suffix}_RESIDUAL_ROBUST.npy', allow_pickle = True)
# RESIDUAL_RANDOM = np.load(f'final_simulation_data/{name}/ac/Robust//{suffix}_RESIDUAL_RANDOM.npy', allow_pickle = True)


# RESIDUAL_ddpg = np.load(f'final_simulation_data/{name}/ac/DDPG_ratio_{test_ratio}/{ddpg_train_type}/{suffix}_RESIDUAL_DRL.npy', allow_pickle = True) #----------------------
# RESIDUAL_td3 = np.load(f'final_simulation_data/{name}/ac/TD3_ratio_{test_ratio}/{td3_train_type}/{suffix}_RESIDUAL_DRL.npy', allow_pickle = True) #----------------------






# RANDOM_ddpg_4 = np.load(f'final_simulation_data/{name}/ac/DDPG_ratio_{test_ratio}/with_ep_schdlr_ptienc_2_0.001_0.001_dropout_03_first_2_layers/{suffix}_TP_RANDOM_DRL.npy', allow_pickle = True) #----------------------

# RANDOM_td3_4 = np.load(f'final_simulation_data/{name}/ac/TD3_ratio_{test_ratio}/with_ep_schdlr_ptienc_2_0.001_0.001_dropout_03_first_2_layers_old_noise/{suffix}_TP_RANDOM_DRL.npy', allow_pickle = True) #----------------------



# RANDOM_ROBUST_AC = np.load(f'final_simulation_data/{name}/ac/Robust/{suffix}_TP_RANDOM_ROBUST.npy', allow_pickle = True)
# RANDOM_RANDOM_AC = np.load(f'final_simulation_data/{name}/ac/Robust/{suffix}_TP_RANDOM_RANDOM.npy', allow_pickle = True)


# ##############   AC DR ################# 



# import os

# print("test_ratio: ", test_ratio)
# print("suffix: ", suffix)
# print("test_ratio", test_ratio)


# path = f"final_simulation_data/case14/ac/figure/AC_{name}"
# if not os.path.exists(path):
#     os.makedirs(path)
# fig = plt.figure(figsize=(8, 6))
# plt.plot(RANDOM_ddpg_4[:-1], color = plt.cm.tab10(3), linestyle = '-', marker = 's', lw = 2.5)
# plt.plot(RANDOM_td3_4[:-1], color = plt.cm.tab10(0), linestyle = '-', marker = 's', lw = 2.5)
# plt.plot(RANDOM_ROBUST_AC[:-1], color = plt.cm.tab10(2), linestyle = '-', marker = 's', lw = 2.5)
# plt.plot(RANDOM_RANDOM_AC[:-1], color = plt.cm.tab10(1), linestyle = '-', marker = 's', lw = 2.5)




# plt.xlabel(r"Attack strength $\rho$")
# plt.ylabel("Attack detection probability")
# plt.xticks(range(len(attack_strength_under_test)), x_axis_name)
# plt.grid()
# plt.legend(['DDPG', 'TD3','Iterative-based Optimization', "Max-Rank"], labelspacing=0.1, loc = 4, fontsize='xx-small')
# plt.ylim([0,1.1])
# # plt.plot()plt.legend(fontsize='xx-small')  # 
# plt.show()

# fig.savefig(f'{path}/DR_rand_Att_ratio_{test_ratio}_{placement}_DDPG_TD3_Robust_MR.pdf', bbox_inches="tight", dpi=400)
# plt.close(fig)



















# import numpy as np

# # ---- Load your robust code results
# # Adjust path as needed
# name = 'case57'  # or 'case57'
# ratio_choice = 3
# col_choice = 0
# dfacts_choise = 0
# suffix = f'choice_{dfacts_choise}_ratio_{ratio_choice}_column_{col_choice}'

# RESIDUAL_ROBUST = np.load(
#     f'final_simulation_data/{name}/ac/Robust/{suffix}_RESIDUAL_ROBUST.npy',
#     allow_pickle=True
# )

# # ---- Attack strength bins (adjust if needed)
# attack_strength_under_test = [5, 7, 10, 15, 20]
# bin_labels = [f"[{attack_strength_under_test[i]}-{attack_strength_under_test[i+1]})"
#               for i in range(len(attack_strength_under_test)-1)]

# print(f"\nRobust Code: Number of Attacks Per Strength Bin for {name} ({suffix}):\n")
# for i, bin_label in enumerate(bin_labels):
#     # Each bin is a list or array of residuals for that strength category
#     if i < len(RESIDUAL_ROBUST):
#         num_attacks = len(RESIDUAL_ROBUST[i])
#         print(f"  {bin_label}: {num_attacks} attacks")
#     else:
#         print(f"  {bin_label}: (No data)")






































import numpy as np
import os

# --------- CONFIGURE YOUR PATHS AND FILE NAMES HERE ---------------
name = 'case57'  # or 'case14'
suffix = "choice_0_ratio_3_column_0"  # adjust as needed
path = f"final_simulation_data/{name}/ac/Robust/"

attack_strength_under_test = [5,7,10,15,20]
x_axis_name = [f"[{attack_strength_under_test[i]}-{attack_strength_under_test[i+1]})" if i < len(attack_strength_under_test)-1 else f"[{attack_strength_under_test[i]}+)" for i in range(len(attack_strength_under_test))]
# ------------------------------------------------------------------

def load_and_describe(filename, bin_labels=None, show_values=False):
    if not os.path.exists(filename):
        print(f"❌ File missing: {filename}")
        return None
    arr = np.load(filename, allow_pickle=True)
    print(f"✅ Loaded: {os.path.basename(filename)} | Type: {type(arr)} | Shape: {arr.shape if hasattr(arr,'shape') else 'n/a'}")
    # If it's a list of lists, print how many elements per bin
    if isinstance(arr, np.ndarray) and arr.dtype == 'O':
        arr = arr.tolist()
    if isinstance(arr, list):
        print("  [Bin counts]:")
        for i, bin_data in enumerate(arr):
            # Handle case where bin_data is not a list/array (e.g., float)
            if hasattr(bin_data, '__len__'):
                n = len(bin_data)
            else:
                n = 1 if bin_data is not None else 0
            label = bin_labels[i] if bin_labels and i < len(bin_labels) else f"Bin {i}"
            summary = f"{bin_data[:3]}..." if show_values and hasattr(bin_data, '__getitem__') and n>0 else ""
            print(f"    {label}: {n} attacks {summary}")
    elif isinstance(arr, np.ndarray) and arr.ndim == 1:
        print("  [First 5 values]:", arr[:5])
    else:
        print("  [Preview]:", str(arr)[:120])
    print()
    return arr

print(f"\n--- Robust Code Output Diagnostic: {name} ({suffix}) ---\n")

# Check all key robust output files (adjust/expand as needed)
files_to_check = [
    f"{path}/{suffix}_RESIDUAL_ATTACKER_ROBUST.npy",
    f"{path}/{suffix}_RESIDUAL_ROBUST.npy",
    f"{path}/{suffix}_RESIDUAL_RANDOM.npy",
    f"{path}/{suffix}_TP_RANDOM_ROBUST.npy",
    f"{path}/{suffix}_TP_RANDOM_RANDOM.npy",
    f"{path}/{suffix}_COST_pre.npy",
    f"{path}/{suffix}_COST_after.npy",
]

# Attack bin labels for pretty printing
bin_labels = [
    "[5-7)",
    "[7-10)",
    "[10-15)",
    "[15-20)",
    "[20+)",
    "(extra bin)"
]

for file in files_to_check:
    print(f"Checking: {file}")
    load_and_describe(file, bin_labels=bin_labels, show_values=False)

print("\nDone. If any bins above have zero attacks and you expected non-zero, the code likely did not finish or generate attacks for those bins.")
