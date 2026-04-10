#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from configs.nn_setting import agent_config 
from utils.settings import *
import configs.config  # or the relevant module
import importlib
importlib.reload(configs.config)

# maxRank_paths
from configs.config import train_config

alpha = agent_config['alpha']
beta = agent_config['beta']

ddpg_path = f"Output_ddpg/{train_config['d-facts_index'][dfacts_choise]}/gma_90_L1_no_schdlr_LN_newR/Testing_ratio_{train_config['test_ratio']}/ddpg/actions/actions_history.csv"
ddpg_path = f"Output_ddpg/{train_config['d-facts_index'][dfacts_choise]}/gma_90_L1_no_schdlr_LN_newR/Testing_ratio_{train_config['test_ratio']}/ddpg/actions/actions_history.csv"
# td3_path = f"Output_td3/{train_config['d-facts_index'][dfacts_choise]}/gma_90_L1_no_schdlr_LN_newR/Testing_ratio_{train_config['test_ratio']}/td3/alpha_{alpha}_beta_{beta}/actions/actions_history.csv"


# ddpg_path = f"Output_ddpg/{train_config['d-facts_index'][dfacts_choise]}/gma_90_L1_no_schdlr_newR/Testing_ratio_{train_config['test_ratio']}/ddpg/actions/actions_history.csv"
# td3_path = f"Output_td3/{train_config['d-facts_index'][dfacts_choise]}/gma_90_L1_no_schdlr_newR/Testing_ratio_{train_config['test_ratio']}/td3/alpha_{alpha}_beta_{beta}/actions/actions_history.csv"

ddpg_4 = f'Output/ddpg/perturb_all_lines_ratio_0.3/with_ep_schdlr_ptienc_2_0.001_0.001_dropout_03_first_2_layers/Testing/ratio_0.5/actions_history.csv'
ddpg_1 = f'Output/ddpg/perturb_all_lines_ratio_0.3/no_schdlr_0.0005_0.0005_copy/Testing/ratio_0.5/actions_history.csv'
ddpg_2 = f'Output/ddpg/perturb_all_lines_ratio_0.3/no_schdlr_0.0005_0.0005_dropout_all_03/Testing/ratio_0.5/actions_history.csv'
ddpg_3 = f'Output/ddpg/perturb_all_lines_ratio_0.3/with_ep_schdlr_ptienc_2_0.001_0.001/Testing/ratio_0.5/actions_history.csv'


# file_path = 'action_history.csv'  # Update with your actual CSV file path
action_data = pd.read_csv(ddpg_4, header=None)
# action_data = action_data[-10:]
# Optional: Check the shape of the data to verify
print(action_data.shape)  # Each row is an episode, each column is an action value in that episode

# Prepare data for plotting
# Flatten the DataFrame for Seaborn; convert episodes and action values into long format
episode_nums = np.repeat(range(1, action_data.shape[0] + 1), action_data.shape[1])
action_values = action_data.values.flatten()

# Plot the action distribution using Seaborn's violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x=episode_nums, y=action_values, inner='quartile', bw=0.2)

# Customize the plot
plt.xlabel('sanple')
plt.ylabel('Action Value')
plt.title('Action Value Distribution over samples')

# Show the plot

plt.show()

