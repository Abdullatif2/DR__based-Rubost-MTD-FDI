#%%
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import importlib
import configs.config  # or the relevant module
importlib.reload(configs.config)

# maxRank_paths
from configs.config import train_config, paths_config #, mtd_config


  

# Select the appropriate paths based on the agent type
selected_paths = paths_config[train_config['agent_type']]

selected_path_ddpg = paths_config['DDPG']
selected_paths_td3 = paths_config['TD3']


# Load the testing files -> DRL based on agent_type (DDPG/TD3)
ratio = train_config['ratio']
print("train ratio = ", ratio)

test_ratio = train_config['test_ratio']
print("test ratio = ", test_ratio)




# plot training rewards 

# fpath_ddpg_score_csv = selected_path_ddpg['training_score_plot_dir']
# fpath_ddpg_score_npy = selected_path_ddpg['training_score_plot_dir']
# fpath_td3_score_csv = selected_paths_td3['training_score_plot_dir']
# fpath_td3_score_npy = selected_paths_td3['training_score_plot_dir']

# saved dir

# test_ratio = train_config['test_ratio']




# DR_file_names = ['DRL', 'BDD', 'Robust'] #, 'Max_Rank']

# Plot configurations
rc = {"font.family": "serif", "mathtext.fontset": "stix"}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
font = {'size': 25}
plt.rc('font', **font)



#%% plot training rewards (different learning rate) (change the output_dir variable)
importlib.reload(configs.config)

from configs.config import train_config 
ratio = train_config['ratio']
print("train ratio =", ratio)
print(f"traing score for ratio = {ratio}")


import numpy as np
import os
import matplotlib.pyplot as plt

# Load the score history data from the specified .npy files
def load_score_history(file_path):
    if os.path.exists(file_path):
        # Load the .npy file
        score_history = np.load(file_path, allow_pickle=True)
        print(f"Loaded score history from {file_path}")
        return score_history
    else:
        print(f"File {file_path} does not exist.")
        return None
        




ddpg_4 = f'Output/ddpg/perturb_all_lines_ratio_0.3/with_ep_schdlr_ptienc_2_0.001_0.001_dropout_03_first_2_layers/score_plot/score_history.npy'
# ddpg_4_score = load_score_history(ddpg_4)

td3_4_n = f'Output/td3/perturb_all_lines_ratio_0.3/with_ep_schdlr_ptienc_2_0.001_0.001_dropout_03_first_2_layers/score_plot/score_history.npy'
td3_4_n_score = load_score_history(td3_4_n)



# Create a figure and axis for the plot
fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(ddpg_4_score, color=plt.cm.tab10(3), linestyle='-', lw=2.5, label="DDPG")
ax.plot(td3_4_n_score, color=plt.cm.tab10(0), linestyle='-', lw=2.5, label="TD3")



# Customize the plot with labels, legend, and grid

# ax.set_title(f"Training Reward for Ratio = {train_config['ratio']}")
ax.set_xlabel('Episodes')
ax.set_ylabel('Reward')
ax.grid(True)
# ax.legend()
ax.legend(loc='upper right')
# Display the plot
plt.show()

# Save the plot as a .pdf file
output_dir = f"final_simulation_data/case57/ac/figure/training"
os.makedirs(output_dir, exist_ok=True)
fig.savefig(f"{output_dir}/temp_FP_reward_plot_{train_config['ratio']}_.pdf", bbox_inches="tight", dpi=400)

# Close the figure to reset the plotting state
plt.close(fig)



#%% lossesssssssssssssssssssssssssssssssssssssssssssss




# import pickle
# import matplotlib.pyplot as plt



# def fill_none_with_previous(data):
#     with open(data, 'rb') as file:
#         data = pickle.load(file)

#     filled_data = []
#     previous_value = 0  # You can change this to the appropriate initial value
    
#     for value in data:
#         if value is None:
#             filled_data.append(previous_value)
#         else:
#             filled_data.append(value)
#             previous_value = value
            
#     return filled_data


# def plot_loss(data_list, batch_size, model, labels, window_size=50):
#     ylim = None        
#     rc = {"font.family" : "serif", 
#         "mathtext.fontset" : "stix",
#         }
#     plt.rcParams.update(rc)
#     plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
#     font = {'size'   : 25}
#     plt.rc('font', **font)
#     fig, ax = plt.subplots(figsize=(8, 6))

#     # Iterate through each dataset in data_list
#     for i, data in enumerate(data_list):
#         # print(len(data))
#         # data = data[:20000]
       
#         data_ = []
#         for j in range(0, len(data), batch_size):
#             batch = data[j:j + batch_size]
            
#             # Check if this is the first batch
#             if j == 0 and len(batch) > initial_discard_size:
#                 # Discard the first initial_discard_size values only from the first batch
#                 processed_batch = batch[initial_discard_size:]
#                 data_.extend(processed_batch)
#             else:
#                 processed_batch = batch[discard_size:]
#                 data_.extend(processed_batch)

#         x = np.arange(len(data_)) / (batch_size-100)  # Create an array for the x-axis (iterations)
#         if i ==0:
#             k = 3
#         else:
#             k = 0
#         if model == 'actor':
#             plt.plot(x, data_,  color=f'C{k}', label=f'{labels[i]}')  # Different color for each dataset
#         else:
            
            
#             plt.plot(x, data_, color=f'C{k}', alpha=0.3)#, label=f'{labels[i]}')
            
#             # Smooth the data using a moving average
#             smoothed_data = np.convolve(data_, np.ones(window_size) / window_size, mode='valid')
#             x_smoothed = np.arange(len(smoothed_data)) / (batch_size-100) 

#             # Plot the smoothed data
#             plt.plot(x_smoothed, smoothed_data, color=f'C{k}', label=f'{labels[i]}')    
#     # Set the limits for the y-axis
#     # plt.ylim([min(min(d) for d in data_list) - 0.5, max(max(d) for d in data_list) + 0.5])


#     # Set y-axis limits
#     if ylim is None:
#         ylim = [min(min(d) for d in data_list) - 0.5, max(max(d) for d in data_list) + 0.5]
#     plt.ylim(ylim)


#     # Add labels, legend, and title
#     plt.xlabel('Episodes')
#     plt.ylabel(f'{model} Loss')
#     plt.title(f'{model.capitalize()} Loss')
#     plt.grid(True)
#     plt.legend()
#     # plt.legend(fontsize='xx-small')  # 
#     # Show the plot
#     plt.show()

#     # Save the plot as a .pdf file
#     output_dir = f"final_simulation_data/case14/ac/figure/training/"
#     os.makedirs(output_dir, exist_ok=True)
#     fig.savefig(f"{output_dir}/{model}_Loss_Comparison_FP_.pdf", bbox_inches="tight", dpi=400)
#     plt.close(fig)



# ddpg_4 = f'Output/ddpg/perturb_all_lines_ratio_0.3/with_ep_schdlr_ptienc_2_0.001_0.001_dropout_03_first_2_layers/model'
# ddpg_4_fr = f'Output/ddpg/minimum_full_rank_ratio_0.3/with_ep_schdlr_ptienc_2_0.001_0.001_dropout_03_first_2_layers/model'

# ddpg_4_actor = fill_none_with_previous(f"{ddpg_4}/actor_losses.pkl")
# ddpg_4_critic = fill_none_with_previous(f"{ddpg_4}/critic_losses.pkl")
# ddpg_4_actor_fr = fill_none_with_previous(f"{ddpg_4_fr}/actor_losses.pkl")
# ddpg_4_critic_fr = fill_none_with_previous(f"{ddpg_4_fr}/critic_losses.pkl")

# td3_4_n = f'Output/td3/perturb_all_lines_ratio_0.3/with_ep_schdlr_ptienc_2_0.001_0.001_dropout_03_first_2_layers_old_noise/model'
# td3_4_n_fr = f'Output/td3/minimum_full_rank_ratio_0.3/with_ep_schdlr_ptienc_2_0.001_0.001_dropout_03_first_2_layers_old_noise/model'

# td3_4_n_actor = fill_none_with_previous(f"{td3_4_n}/actor_losses.pkl")
# td3_4_n_critic = fill_none_with_previous(f"{td3_4_n}/critic_losses.pkl")
# td3_4_n_actor_fr = fill_none_with_previous(f"{td3_4_n_fr}/actor_losses.pkl")
# td3_4_n_critic_fr = fill_none_with_previous(f"{td3_4_n_fr}/critic_losses.pkl")


# batch_size = 500
# initial_discard_size = 128
# discard_size = 100

# model = "Actor"
# model = "Critic"

# actor_losses = [ddpg_4_actor, td3_4_n_actor]
# actor_labels = ['DDPG','TD3']
# critic_losses = [ddpg_4_critic, td3_4_n_critic]
# critic_labels = ['DDPG','TD3']

# # actor_losses = [ddpg_4_actor_fr, td3_4_n_actor_fr]
# # actor_labels = ['DDPG','TD3']
# # critic_losses = [ddpg_4_critic_fr, td3_4_n_critic_fr]
# # critic_labels = ['DDPG','TD3']


# # Plot multiple actor losses
# plot_loss(actor_losses, 500, 'actor', actor_labels)
# plot_loss(critic_losses, 500, 'critic', critic_labels)







#%%
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os


def fill_none_with_previous(filepath):
    with open(filepath, 'rb') as file:
        data = pickle.load(file)

    filled_data = []
    previous_value = 0  # Default value for the first `None` if encountered
    
    for value in data:
        if value is None:
            filled_data.append(previous_value)
        else:
            filled_data.append(value)
            previous_value = value
            
    return filled_data


def plot_loss(data_list, batch_size, model, labels, window_size=50, ylim=None):
    rc = {
        "font.family": "serif",
        "mathtext.fontset": "stix",
    }
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    font = {'size': 25}
    plt.rc('font', **font)
    
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, data in enumerate(data_list):
        data_ = []
        for j in range(0, len(data), batch_size):
            batch = data[j:j + batch_size]

            # Process the batches based on discard parameters
            if j == 0 and len(batch) > initial_discard_size:
                processed_batch = batch[initial_discard_size:]
            else:
                processed_batch = batch[discard_size:]

            data_.extend(processed_batch)

        x = np.arange(len(data_)) / (batch_size - 100)  # X-axis in terms of iterations
        
        k = 3 if i == 0 else 0  # Assign different colors for each dataset
        
        # Plot raw data
        # plt.plot(x, data_, color=f'C{k}', alpha=0.3 if model == 'critic' else 1)#, label=f'{labels[i]}')
        # plt.legend(loc='upper right')  # Move legend to the top-right corner


        # # Apply smoothing only for critic model
        # if model == 'critic':
        #     smoothed_data = np.convolve(data_, np.ones(window_size) / window_size, mode='valid')
        #     x_smoothed = np.arange(len(smoothed_data)) / (batch_size - 100)
        #     plt.plot(x_smoothed, smoothed_data, color=f'C{k}', label=f'{labels[i]} ')
        #     plt.legend(loc='lower right')  # Move legend to the top-right corner


        if model == 'actor':
            # For actor: plot the line and give it a label
            plt.plot(x, data_, color=f'C{k}', label=f'{labels[i]}')
        else:  # critic
            # For critic: plot raw (faint) + smoothed (with label)
            plt.plot(x, data_, color=f'C{k}', alpha=0.3)
            smoothed_data = np.convolve(data_, np.ones(window_size) / window_size, mode='valid')
            x_smoothed = np.arange(len(smoothed_data)) / (batch_size - 100)
            plt.plot(x_smoothed, smoothed_data, color=f'C{k}', label=f'{labels[i]}')

        plt.legend(loc='upper right')



    # Set y-axis limits
    if ylim is None:
        ylim = [min(min(d) for d in data_list) - 0.5, max(max(d) for d in data_list) + 0.5]
    plt.ylim(ylim)

    # Add labels, title, and legend
    plt.xlabel('Episodes')
    plt.ylabel(f'{model.capitalize()} Loss')
    # plt.title(f'{model.capitalize()} Loss')
    plt.grid(True)

    # Save and show the plot
    output_dir = f"final_simulation_data/case57/ac/figure/training/"
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f"{output_dir}/{model}_Loss_Comparison_FP_.pdf", bbox_inches="tight", dpi=400)
    plt.show()
    plt.close(fig)


# Load and process data
ddpg_4 = f'Output/ddpg/perturb_all_lines_ratio_0.3/with_ep_schdlr_ptienc_2_0.001_0.001_dropout_03_first_2_layers/model'
ddpg_4_actor = fill_none_with_previous(f"{ddpg_4}/actor_losses.pkl")
ddpg_4_critic = fill_none_with_previous(f"{ddpg_4}/critic_losses.pkl")

td3_4_n = f'Output/td3/perturb_all_lines_ratio_0.3/with_ep_schdlr_ptienc_2_0.001_0.001_dropout_03_first_2_layers/model'
td3_4_n_actor = fill_none_with_previous(f"{td3_4_n}/actor_losses.pkl")
td3_4_n_critic = fill_none_with_previous(f"{td3_4_n}/critic_losses.pkl")


batch_size = 500
initial_discard_size = 128
discard_size = 100

# Combine actor and critic loss data
actor_losses = [ddpg_4_actor, td3_4_n_actor]
critic_losses = [ddpg_4_critic, td3_4_n_critic]
actor_labels = ['DDPG', 'TD3']
critic_labels = ['DDPG', 'TD3']



# Compute shared ylim
shared_ylim = [
    min(min(actor_losses[0]), min(actor_losses[1]), min(critic_losses[0]), min(critic_losses[1])) - 0.5,
    max(max(actor_losses[0]), max(actor_losses[1]), max(critic_losses[0]), max(critic_losses[1])) + 0.5,
]

# Plot
plot_loss(actor_losses, batch_size, 'actor', actor_labels, ylim=shared_ylim)
plot_loss(critic_losses, batch_size, 'critic', critic_labels, ylim=shared_ylim)
