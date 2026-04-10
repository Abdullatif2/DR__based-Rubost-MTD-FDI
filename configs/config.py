
#%% saving for running new scaled reward (FP)
# from utils.settings import *

from pypower.api import ppoption
import numpy as np
from configs.nn_setting import agent_config 
from utils.settings import *
from utils.initialization import *

alpha = agent_config['alpha']
beta = agent_config['beta']

# case_name = 'case14'
case_name = 'case57'


train_config = { 
    'seed_value': 42,
    'mode': 'test',  # 'train' or 'test'    
    'batch_size': 500, # iterations
    'episodes': 500, # of training episodes 
    'ele_remove': 100, # of elements removed in each episode
    'ratio': 0.3, # training perturbation ratio 
    'agent_type': 'TD3',  # Can be 'DDPG' or 'TD3'
    'test_ratio': x_max_ratio, # testing perturbation ratio  
    'changes_type':f"with_ep_schdlr_ptienc_2_{alpha}_{beta}_dropout_03_first_2_layers", #----- (7) 

    'd-facts_index': {
        0: 'perturb_all_lines',
        1: 'minimum_full_rank',
        2: 'minimum'},
}   



ddpg_paths = {
    # training 
    'training_score_plot_dir': f"Output/ddpg/{train_config['d-facts_index'][dfacts_choise]}_ratio_{train_config['ratio']}/{train_config['changes_type']}/score_plot",
    'actor_model_dir': f"Output/ddpg/{train_config['d-facts_index'][dfacts_choise]}_ratio_{train_config['ratio']}/{train_config['changes_type']}/model",
    'critic_model_dir': f"Output/ddpg/{train_config['d-facts_index'][dfacts_choise]}_ratio_{train_config['ratio']}/{train_config['changes_type']}/model",
    'model_losses': f"Output/ddpg/{train_config['d-facts_index'][dfacts_choise]}_ratio_{train_config['ratio']}/{train_config['changes_type']}/model",
   
    # testing 
    'testing_score_plot_dir': f"Output/ddpg/{train_config['d-facts_index'][dfacts_choise]}_ratio_{train_config['ratio']}/{train_config['changes_type']}/Testing/ratio_{train_config['test_ratio']}",
    'testing_time_plot_dir': f"Output/ddpg/{train_config['d-facts_index'][dfacts_choise]}_ratio_{train_config['ratio']}/{train_config['changes_type']}/Testing/ratio_{train_config['test_ratio']}",
    'testing_actions_dir': f"Output/ddpg/{train_config['d-facts_index'][dfacts_choise]}_ratio_{train_config['ratio']}/{train_config['changes_type']}/Testing/ratio_{train_config['test_ratio']}",
}
td3_paths = {
    # training 
    'training_score_plot_dir': f"Output/td3/{train_config['d-facts_index'][dfacts_choise]}_ratio_{train_config['ratio']}/{train_config['changes_type']}/score_plot",
    'actor_model_dir': f"Output/td3/{train_config['d-facts_index'][dfacts_choise]}_ratio_{train_config['ratio']}/{train_config['changes_type']}/model",
    'critic_model_dir': f"Output/td3/{train_config['d-facts_index'][dfacts_choise]}_ratio_{train_config['ratio']}/{train_config['changes_type']}/model",
    'model_losses': f"Output/td3/{train_config['d-facts_index'][dfacts_choise]}_ratio_{train_config['ratio']}/{train_config['changes_type']}/model",
    # testing 
    'testing_score_plot_dir': f"Output/td3/{train_config['d-facts_index'][dfacts_choise]}_ratio_{train_config['ratio']}/{train_config['changes_type']}/Testing/ratio_{train_config['test_ratio']}",
    'testing_time_plot_dir': f"Output/td3/{train_config['d-facts_index'][dfacts_choise]}_ratio_{train_config['ratio']}/{train_config['changes_type']}/Testing/ratio_{train_config['test_ratio']}",
    'testing_actions_dir': f"Output/td3/{train_config['d-facts_index'][dfacts_choise]}_ratio_{train_config['ratio']}/{train_config['changes_type']}/Testing/ratio_{train_config['test_ratio']}",
}



paths_config = {
    'DDPG': ddpg_paths,
    'TD3': td3_paths,
}

