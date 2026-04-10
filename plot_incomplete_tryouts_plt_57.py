
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
# ratio_choice = 1        ## the level of D-FACTS perturbation ratio ---- 0 : pm 0.2 --- 1 : pm 0.3 --- 2 : pm 0.4 --- 3 : pm 0.5
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
# td3_train_type = 'with_ep_schdlr_ptienc_2_0.001_0.001_dropout_03_first_2_layers'

# agent_type = train_config['agent_type']
# test_ratio = x_max_ratio


# # name = 'case14'
# name = 'case57'

# suffix = f'choice_{dfacts_choise}_ratio_{ratio_choice}_column_{col_choice}'

# # if name == 'case14':
# #     BDD_threshold = 58.124
# # elif name == 'case57':
# #     BDD_threshold = 192.700
    
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
# # RESIDUAL = np.load(f'final_simulation_data/{name}/ac/Robust/{suffix}_RESIDUAL_ATTACKER_ROBUST.npy', allow_pickle = True)

# # RANDOM_ddpg_AC = np.load(f'final_simulation_data/{name}/ac/DDPG_ratio_{test_ratio}/{ddpg_train_type}/{suffix}_TP_RANDOM_DRL.npy', allow_pickle = True) #----------------------
# # RESIDUAL_att_ddpg = np.load(f'final_simulation_data/{name}/ac/DDPG_ratio_{test_ratio}/{ddpg_train_type}/{suffix}_RESIDUAL_ATTACKER_DRL.npy', allow_pickle = True) #----------------------

# # RANDOM_td3_AC = np.load(f'final_simulation_data/{name}/ac/TD3_ratio_{test_ratio}/{td3_train_type}/{suffix}_TP_RANDOM_DRL.npy', allow_pickle = True) #----------------------
# # RESIDUAL_att_td3 = np.load(f'final_simulation_data/{name}/ac/TD3_ratio_{test_ratio}/{td3_train_type}/{suffix}_RESIDUAL_ATTACKER_DRL.npy', allow_pickle = True) #----------------------



# # COST_PRE_td3 = np.load(f'final_simulation_data/{name}/ac/TD3_ratio_{test_ratio}/{td3_train_type}/{suffix}_COST_pre.npy', allow_pickle = True) #----------------------
# # COST_AFTER_td3 = np.load(f'final_simulation_data/{name}/ac/TD3_ratio_{test_ratio}/{td3_train_type}/{suffix}_COST_after.npy', allow_pickle = True) #----------------------

# # COST_PRE = np.load(f'final_simulation_data/{name}/ac/{suffix}_COST_pre.npy', allow_pickle = True)
# # COST_AFTER = np.load(f'final_simulation_data/{name}/ac/{suffix}_COST_after.npy', allow_pickle = True)


# # COST_PRE_ddpg = np.load(f'final_imulation_data/{name}/ac/DDPG_ratio_{test_ratio}/{ddpg_train_type}/{suffix}_COST_pre.npy', allow_pickle = True) #----------------------
# # COST_AFTER_ddpg = np.load(f'final_simulation_data/{name}/ac/DDPG_ratio_{test_ratio}/{ddpg_train_type}/{suffix}_COST_after.npy', allow_pickle = True) #----------------------
# # COST_PRE_td3 = np.load(f'final_simulation_data/{name}/ac/TD3_ratio_{test_ratio}/{td3_train_type}/{suffix}_COST_pre.npy', allow_pickle = True) #----------------------
# # COST_AFTER_td3 = np.load(f'final_simulation_data/{name}/ac/TD3_ratio_{test_ratio}/{td3_train_type}/{suffix}_COST_after.npy', allow_pickle = True) #----------------------

# # residual
# # RESIDUAL_ROBUST = np.load(f'final_simulation_data/{name}/ac/Robust//{suffix}_RESIDUAL_ROBUST.npy', allow_pickle = True)
# # RESIDUAL_RANDOM = np.load(f'final_simulation_data/{name}/ac/Robust//{suffix}_RESIDUAL_RANDOM.npy', allow_pickle = True)


# # RESIDUAL_ddpg = np.load(f'final_simulation_data/{name}/ac/DDPG_ratio_{test_ratio}/{ddpg_train_type}/{suffix}_RESIDUAL_DRL.npy', allow_pickle = True) #----------------------
# RESIDUAL_td3 = np.load(f'final_simulation_data/{name}/ac/TD3_ratio_{test_ratio}/{td3_train_type}/{suffix}_RESIDUAL_DRL.npy', allow_pickle = True) #----------------------






# RANDOM_ddpg_4 = np.load(f'final_simulation_data/{name}/ac/DDPG_ratio_{test_ratio}/with_ep_schdlr_ptienc_2_0.001_0.001_dropout_03_first_2_layers/{suffix}_TP_RANDOM_DRL.npy', allow_pickle = True) #----------------------

# RANDOM_td3_4 = np.load(f'final_simulation_data/{name}/ac/TD3_ratio_{test_ratio}/with_ep_schdlr_ptienc_2_0.001_0.001_dropout_03_first_2_layers/{suffix}_TP_RANDOM_DRL.npy', allow_pickle = True) #----------------------



# # RANDOM_ROBUST_AC = np.load(f'final_simulation_data/{name}/ac/Robust/{suffix}_TP_RANDOM_ROBUST.npy', allow_pickle = True)
# # RANDOM_RANDOM_AC = np.load(f'final_simulation_data/{name}/ac/Robust/{suffix}_TP_RANDOM_RANDOM.npy', allow_pickle = True)
# RANDOM_RANDOM_AC = np.load(f'final_simulation_data/{name}/ac/Random/{suffix}_TP_RANDOM_RANDOM.npy', allow_pickle = True)


# ##############   AC DR ################# 



# import os

# print("test_ratio: ", test_ratio)
# print("suffix: ", suffix)
# print("test_ratio", test_ratio)


# # path = f"final_simulation_data/case14/ac/figure/AC_{name}"
# path = f"final_simulation_data/case57/ac/figure_Random/AC_{name}"
# if not os.path.exists(path):
#     os.makedirs(path)
# fig = plt.figure(figsize=(8, 6))
# plt.plot(RANDOM_ddpg_4[:-1], color = plt.cm.tab10(3), linestyle = '-', marker = 's', lw = 2.5)
# plt.plot(RANDOM_td3_4[:-1], color = plt.cm.tab10(0), linestyle = '-', marker = 's', lw = 2.5)
# # plt.plot(RANDOM_ROBUST_AC[:-1], color = plt.cm.tab10(2), linestyle = '-', marker = 's', lw = 2.5)
# plt.plot(RANDOM_RANDOM_AC[:-1], color = plt.cm.tab10(1), linestyle = '-', marker = 's', lw = 2.5)




# plt.xlabel(r"Attack strength $\rho$")
# plt.ylabel("Attack detection probability")
# plt.xticks(range(len(attack_strength_under_test)), x_axis_name)
# plt.grid()
# # plt.legend(['DDPG', 'TD3','Iterative-based Optimization', "Max-Rank"], labelspacing=0.1, loc = 4, fontsize='xx-small')
# plt.legend(['DDPG', 'TD3', "Max-Rank"], labelspacing=0.1, loc = 4, fontsize='xx-small')
# plt.ylim([0,1.1])
# # plt.plot()plt.legend(fontsize='xx-small')  # 
# plt.show()

# fig.savefig(f'{path}/DR_rand_Att_ratio_{test_ratio}_{placement}_DDPG_TD3_MR.pdf', bbox_inches="tight", dpi=400)
# plt.close(fig)





























































































# # %%

# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn
# from scipy.stats import ncx2, chi2
# from configs.config import train_config, paths_config
# import os



# # % SETTINGS

# grid_choice = 1         ## 0:case6, 1:case14, 2:case57
# ratio_choice = 1        ## the level of D-FACTS perturbation ratio ---- 0 : pm 0.2 --- 1 : pm 0.3 --- 2 : pm 0.4 --- 3 : pm 0.5
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
# td3_train_type = 'with_ep_schdlr_ptienc_2_0.001_0.001_dropout_03_first_2_layers'

# agent_type = train_config['agent_type']
# test_ratio = x_max_ratio


# # name = 'case14'
# name = 'case57'

# suffix = f'choice_{dfacts_choise}_ratio_{ratio_choice}_column_{col_choice}'

# # if name == 'case14':
# #     BDD_threshold = 58.124
# # elif name == 'case57':
# #     BDD_threshold = 192.700
    
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

# RANDOM_ddpg_4 = np.load(f'final_simulation_data/{name}/ac/DDPG_ratio_{test_ratio}/with_ep_schdlr_ptienc_2_0.001_0.001_dropout_03_first_2_layers/{suffix}_TP_RANDOM_DRL.npy', allow_pickle = True) #----------------------
# # RANDOM_td3_4 = np.load(f'final_simulation_data/{name}/ac/TD3_ratio_{test_ratio}/with_ep_schdlr_ptienc_2_0.001_0.001_dropout_03_first_2_layers/{suffix}_TP_RANDOM_DRL.npy', allow_pickle = True) #----------------------
# RANDOM_ROBUST_AC = np.load(f'final_simulation_data/{name}/ac/Robust/{suffix}_TP_RANDOM_ROBUST.npy', allow_pickle = True)
# RANDOM_RANDOM_AC = np.load(f'final_simulation_data/{name}/ac/Random/{suffix}_TP_RANDOM_RANDOM.npy', allow_pickle = True)
# RANDOM_td3_4 = np.load(f'final_simulation_data/{name}/ac/TD3_ratio_0.3_check_td3_or_ddpg/with_ep_schdlr_ptienc_2_0.001_0.001_dropout_03_first_2_layers/{suffix}_TP_RANDOM_DRL.npy', allow_pickle = True) #----------------------

# ##############   AC DR ################# 

# print("test_ratio: ", test_ratio)
# print("suffix: ", suffix)
# print("test_ratio", test_ratio)


# path = f"final_simulation_data/case57/ac/figure_all/AC_{name}"
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
# plt.legend(['DDPG', 'TD3','Iterative-based Optimization',  "Max-Rank"], labelspacing=0.1, loc = 4, fontsize='xx-small')
# plt.ylim([0,1.1])
# plt.show()

# fig.savefig(f'{path}/DR_rand_Att_ratio_{test_ratio}_{placement}_DDPG_TD3_MR_2.pdf', bbox_inches="tight", dpi=400)
# plt.close(fig)





















# # %%
# import numpy as np
# import os

# # --- reuse your settings ---
# name = 'case57'
# ratio_choice = 1
# dfacts_choise = 0
# col_choice = 0
# ddpg_train_type = 'with_ep_schdlr_ptienc_2_0.001_0.001_dropout_03_first_2_layers'

# # derive test_ratio and suffix exactly like your code
# test_ratio = {0:0.2, 1:0.3, 2:0.4, 3:0.5}.get(ratio_choice, 0.3)
# suffix = f'choice_{dfacts_choise}_ratio_{ratio_choice}_column_{col_choice}'

# ddpg_path = (
#     f'final_simulation_data/{name}/ac/DDPG_ratio_{test_ratio}/'
#     f'{ddpg_train_type}/{suffix}_TP_RANDOM_DRL.npy'
# )

# print("Loading:", ddpg_path)
# if not os.path.exists(ddpg_path):
#     raise FileNotFoundError(f"File not found: {ddpg_path}")

# arr = np.load(ddpg_path, allow_pickle=True)

# # Print raw array
# np.set_printoptions(precision=4, suppress=True)
# print("\nDDPG full array:")
# print(arr)

# # What you actually plotted ([:-1])
# print("\nDDPG array used in plot ([:-1]):")
# print(arr[:-1])

# # Quick diagnostics
# print("\nDiagnostics:")
# print("shape:", arr.shape, "dtype:", arr.dtype)
# print("min:", np.min(arr), "max:", np.max(arr), "mean:", float(np.mean(arr)))
# print("all zeros?", bool(np.all(arr == 0)))
# print("all zeros in plotted slice ([:-1])?", bool(np.all(arr[:-1] == 0)))
# print("unique values (first 10):", np.unique(arr)[:10])

# # If you expect 5 points (for rho = [5,7,10,15,20]), check length:
# print("\nExpected 5 points? length of arr[:-1] =", len(arr[:-1]))












# %%

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from scipy.stats import ncx2, chi2
from configs.config import train_config, paths_config
import os



# % SETTINGS

grid_choice = 1         ## 0:case6, 1:case14, 2:case57
ratio_choice = 1        ## the level of D-FACTS perturbation ratio ---- 0 : pm 0.2 --- 1 : pm 0.3 --- 2 : pm 0.4 --- 3 : pm 0.5
dfacts_choise = 0       ## the d-facts perturbation ----- 0 : perturb all lines --- 1 : minimum full rank --- 2 : minimum 
col_choice = 0          ## the on/off of the column angle constraint ---- 0 : no column constraint 1 : with column constraint

if ratio_choice == 0:
    ratio = 0.2
    x_max_ratio = 0.2 ## |delta x| <= x_max_ratio
elif ratio_choice == 1:
    ratio = 0.3
    x_max_ratio = 0.3 ## |delta x| <= x_max_ratio
elif ratio_choice == 2:
    ratio = 0.4
    x_max_ratio = 0.4 ## |delta x| <= x_max_ratio
elif ratio_choice == 3:
    ratio = 0.5 # the maximum in liturature
    x_max_ratio = 0.5 ## |delta x| <= x_max_ratio

elif ratio_choice == 10:
    # only for a test
    ratio = 0.1
    x_max_ratio = 0.1  ## |delta x| <= x_max_ratio
    x_min_ratio = 0.01 ## |delta_x|

placement = ''
if dfacts_choise == 0:
    placement = 'FP'
else:
    placement = 'FR'


ddpg_train_type = 'with_ep_schdlr_ptienc_2_0.001_0.001_dropout_03_first_2_layers'
td3_train_type = 'with_ep_schdlr_ptienc_2_0.001_0.001_dropout_03_first_2_layers'

agent_type = train_config['agent_type']
test_ratio = x_max_ratio


# name = 'case14'
name = 'case57'

suffix = f'choice_{dfacts_choise}_ratio_{ratio_choice}_column_{col_choice}'

# if name == 'case14':
#     BDD_threshold = 58.124
# elif name == 'case57':
#     BDD_threshold = 192.700
    
rc = {"font.family" : "serif", 
      "mathtext.fontset" : "stix",
      }
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
font = {'size'   : 25}
plt.rc('font', **font)

attack_strength_under_test = [5,7,10,15,20]

x_axis_name = [str(i) for i in attack_strength_under_test]



# % LOAD DATA

# RANDOM_ddpg_4 = np.load(f'ffinal_simulation_data/{name}/ac/DDPG_ratio_{test_ratio}/with_ep_schdlr_ptienc_2_0.001_0.001_dropout_03_first_2_layers/{suffix}_TP_RANDOM_DRL.npy', allow_pickle = True) #----------------------
# RANDOM_td3_4 = np.load(f'ffinal_simulation_data/{name}/ac/TD3_ratio_{test_ratio}/with_ep_schdlr_ptienc_2_0.001_0.001_dropout_03_first_2_layers/{suffix}_TP_RANDOM_DRL.npy', allow_pickle = True) #----------------------
# RANDOM_ROBUST_AC = np.load(f'ffinal_simulation_data/{name}/ac/Robust/{suffix}_TP_RANDOM_ROBUST.npy', allow_pickle = True)
# RANDOM_RANDOM_AC = np.load(f'ffinal_simulation_data/{name}/ac/Random/{suffix}_TP_RANDOM_RANDOM.npy', allow_pickle = True)
# # RANDOM_td3_4 = np.load(f'ffinal_simulation_data/{name}/ac/TD3_ratio_0.3_check_td3_or_ddpg/with_ep_schdlr_ptienc_2_0.001_0.001_dropout_03_first_2_layers/{suffix}_TP_RANDOM_DRL.npy', allow_pickle = True) #----------------------


ddpg_Time = np.load(f'ffinal_simulation_data/{name}/ac/DDPG_ratio_{test_ratio}/w/{suffix}_TP_RANDOM_DRL.npy', allow_pickle = True) #----------------------
td3_Time = np.load(f'ffinal_simulation_data/{name}/ac/TD3_ratio_{test_ratio}/w/{suffix}_TP_RANDOM_DRL.npy', allow_pickle = True) #----------------------
td3_Time = np.load(f'ffinal_simulation_data/{name}/ac/TD3_ratio_{test_ratio}/w/{suffix}_TP_RANDOM_DRL.npy', allow_pickle = True) #----------------------


ROBUST_Time = np.load(f'ffinal_simulation_data/{name}/ac/Robust/{suffix}_TIME_ROBUST.npy', allow_pickle = True)

print("ROBUST_Time: ", ROBUST_Time)
print("td3_Time: ", td3_Time)
print("ddpg_Time: ", ddpg_Time)











##############   AC DR ################# 

# print("test_ratio: ", test_ratio)
# print("suffix: ", suffix)
# print("test_ratio", test_ratio)


# path = f"ffinal_simulation_data/case57/ac/figure_all/AC_{name}"
# if not os.path.exists(path):
#     os.makedirs(path)
# fig = plt.figure(figsize=(8, 6))
# # plt.plot(RANDOM_ddpg_4[:-1], color = plt.cm.tab10(3), linestyle = '-', marker = 's', lw = 2.5)
# plt.plot(RANDOM_td3_4[:-1], color = plt.cm.tab10(0), linestyle = '-', marker = 's', lw = 2.5)
# plt.plot(RANDOM_ROBUST_AC[:-1], color = plt.cm.tab10(2), linestyle = '-', marker = 's', lw = 2.5)
# plt.plot(RANDOM_RANDOM_AC[:-1], color = plt.cm.tab10(1), linestyle = '-', marker = 's', lw = 2.5)




# plt.xlabel(r"Attack strength $\rho$")
# plt.ylabel("Attack detection probability")
# plt.xticks(range(len(attack_strength_under_test)), x_axis_name)
# plt.grid()
# plt.legend(['TD3','Iterative-based Optimization',  "Max-Rank"], labelspacing=0.1, loc = 4, fontsize='xx-small')
# plt.ylim([0,1.1])
# plt.show()

# fig.savefig(f'{path}/DR_rand_Att_ratio_{test_ratio}_{placement}_TD3_MR_Iterative_3.pdf', bbox_inches="tight", dpi=400)
# plt.close(fig)


















