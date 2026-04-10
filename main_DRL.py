import os
import numpy as np
import random
import torch
from utils.initialization import *
from utils.evaluation_fun import generate_data
import pickle
import numpy as np
from configs.config import train_config, paths_config
from configs.nn_setting import nn_config, agent_config
import pandas as pd
import ast
from DRL.PS_env import CustomEnv
from utils_2.save_results import *
from utils_2.custom_dataset import PowerDataDataset
from torch.utils.data import DataLoader
from torch.profiler import profile 

# random.seed(train_config['seed_value'])
# np.random.seed(train_config['seed_value'])
torch.manual_seed(train_config['seed_value'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(train_config['seed_value'])

selected_paths = paths_config[train_config['agent_type']]

# Import the appropriate agent based on the configuration
if train_config['agent_type'] == 'DDPG':
    print("Agent:", train_config['agent_type'])
    from DRL.ddpg_torch import Agent 
elif train_config['agent_type'] == 'TD3':
    print("Agent:", train_config['agent_type'])
    from DRL.td3_torch import Agent 
else:
    raise ValueError("Unsupported agent type specified in the configuration.")

def initialize_agent(agent_type, env, agent_config, nn_config, ratio, input_dims):
    if agent_type != None:
        n_actions = env.dfacts_index.shape[0] #20   
        agent = Agent(
        ratio=ratio,
        alpha=agent_config['alpha'],
        beta = agent_config['beta'],
        input_dims=input_dims,
        tau=0.005,
        n_actions=n_actions,
        Act_fc1_dims=512,
        Act_fc2_dims=256,
        Act_fc3_dims=128,
        Act_fc4_dims=0,  
        Crtc_fc1_dims=512,
        Crtc_fc2_dims=256,
        Crtc_fc3_dims=128,
        Crtc_fc4_dims=0,  
        batch_size=128,
        gamma=0.90,
        max_size=1e6,
        )
    else:
        raise ValueError("Unsupported agent type specified in the configuration.")
    return agent

def eval_policy(agent, case, val_iterations=20):
    val_score = 0
    for itr in range(val_iterations):  # Iterate over the validation dataset
        active_power, reactive_power, v_mag_est, v_ang_est, Jr_N, result = generate_data(case)
        _, _, z_noise = case.ac_opf(active_power, reactive_power)
        state = z_noise
        all_state_var = [active_power, reactive_power, v_mag_est, v_ang_est]
        action, _ = agent.choose_action(state, itr, is_training=False)  # No exploration noise for evaluation
        _, val_reward, _ = env.step(action, all_state_var)
        print(f"val_iterations # {itr+1}/{val_iterations}, iteration reward {val_reward}")  

        val_score += val_reward
    avg_val_score = val_score / val_iterations  # Compute the average validation score
    return avg_val_score


if __name__ == '__main__':
    COST_pre = []
    COST_after = []
    # ac_opf_no_ac = 100   # ----> for loop
    actor_losses = []
    critic_losses = []
    val_score = 0


    observation_space = 2*case.no_bus + 2*case.no_branch 
    print("observation_space: ", observation_space)
    score_history = []
    ele_remove = train_config['ele_remove']

    critic_losses=[]
    actor_losses=[]
    test_score_history = []
    best_score = -np.inf
    average_score = 0
    print("dfacts_index", dfacts_index)



    if train_config['mode'] == 'train':
        # Load the dataset
        # csv_file = 'dataset_case14_training.csv'
        csv_file = 'dataset_case57_training.csv'
        

        train_dataset  = PowerDataDataset(csv_file, start_row=0, num_rows=25000)    # First 25,000 rows for training
        dataloader = DataLoader(train_dataset , batch_size=train_config['batch_size'], shuffle=False)
        # numer of iterations in each episode
        train_iterations = train_config['batch_size']
        
        if not os.path.exists(selected_paths['model_losses']):
            os.makedirs(selected_paths['model_losses'])
        
        episodes = train_config['episodes']
        print("ratio =", train_config['ratio'])
        print("changed learning =", train_config['changes_type'])
        print("total # of episodes =", train_config['episodes'])
        

        env = CustomEnv(case, train_iterations, train_config['ratio'], dfacts_index, observation_space, train_config['mode'])
        agent = initialize_agent(train_config['agent_type'], env, agent_config, nn_config,  train_config['ratio'], observation_space)
        print("env.observation_space.shape", env.observation_space.shape)

        save_interval = 2
        eval_frequency = 3
        data_iter = iter(dataloader)
        for episode in range(episodes):
            try:
                batch = next(data_iter)  # Fetch one new batch
            except StopIteration:
                # If the DataLoader runs out of data, reinitialize the iterator
                data_iter = iter(dataloader)
                batch = next(data_iter)
    
            observation = env.reset()
            agent.time_step = 0
            score = 0
            done = False
            # for iteration in range(train_iterations):  
            for iteration in range(len(batch['active_power'])):
                # Extract each sample from the batch
                active_power = batch['active_power'][iteration]
                reactive_power = batch['reactive_power'][iteration]
                v_mag_est = batch['v_mag_est'][iteration]
                v_ang_est = batch['v_ang_est'][iteration]
                result_f = batch['result_f'][iteration]

                # active_power, reactive_power, v_mag_est, v_ang_est, Jr_N, result = generate_data(case)
                result, z_new, z_noise = case.ac_opf(active_power, reactive_power)
                COST_pre.append(result['f'])
                all_state_var = [active_power, reactive_power, v_mag_est, v_ang_est]
                action, time = agent.choose_action(z_noise, episode, is_training=True) 
                next_state, reward, done = env.step(action, all_state_var)
                if iteration >= ele_remove:
                    score += reward
                agent.remember(z_noise, action, reward, next_state, done)
                actor_loss,critic_loss = agent.learn()
                critic_losses.append(critic_loss)
                actor_losses.append(actor_loss)
 

                print(f"episode = {episode}, iteration # {iteration+1}/{train_iterations}, iteration reward {reward}, Actor Loss: {actor_loss}, Critic Loss: {critic_loss} ", )  
                
            average_score = score / (train_iterations - ele_remove)
            score_history.append(average_score)    

            # scheduler per episode  
            average_actor_loss = np.mean([loss for loss in actor_losses[-100:] if loss is not None])
            average_critic_loss = np.mean([loss for loss in critic_losses[-100:] if loss is not None])
            agent.actor.scheduler.step(average_actor_loss) 
            if train_config['agent_type'] == 'DDPG':
                agent.critic.scheduler.step(average_critic_loss) 
            else:
                agent.critic_1.scheduler.step(average_critic_loss)     
            # Save the model if the validation score improves
            if average_score > best_score:
                best_score = average_score
                print(f"New best score: {best_score}, saving model...")
                agent.save_models()
            
            # saving each episode 
            with open(f"{selected_paths['model_losses']}/critic_losses.pkl", 'wb') as f:
                pickle.dump(critic_losses, f)
            with open(f"{selected_paths['model_losses']}/actor_losses.pkl", 'wb') as f:
                pickle.dump(actor_losses, f)

            # save reward each episode (to save and see results while training) 
            if (episode + 1) % save_interval == 0:
                save_scores(selected_paths['training_score_plot_dir'], score_history)
                print(f"Reward history saved after episode {episode + 1}")
        # Final save after all episodes
        save_scores(selected_paths['training_score_plot_dir'], score_history)



    elif train_config['mode'] == 'test':

        test_size = ac_opf_no_ac
        # csv_file = 'dataset_case14_training.csv'
        csv_file = 'dataset_case57_training.csv'
        
        test_dataset = PowerDataDataset(csv_file, start_row=25000, num_rows=test_size)    # First 25,000 rows for training
        dataloader = DataLoader(test_dataset, batch_size=test_size, shuffle=False)
        data_iter = iter(dataloader)
        batch = next(data_iter)       
        print("test ratio =", train_config['test_ratio'])
        env = CustomEnv(case, test_size, train_config['test_ratio'], dfacts_index, observation_space, train_config['mode'])
        agent = initialize_agent(train_config['agent_type'], env, agent_config, nn_config, train_config['test_ratio'], observation_space)
        agent.load_models()
        
        
        # state = env.reset(all_state_var) 
        test_score = 0
        done = False
        actions_history = []
        TIME_DRL = []
    




        # state = env.reset(all_state_var) 
        test_score = 0
        done = False
        actions_history = []
        TIME_DRL = []
        # Now iterate over each sample within the batch
        for i in range(test_size):
            # Extract each sample from the batch dictionary
            active_power = batch['active_power'][i]
            reactive_power = batch['reactive_power'][i]
            v_mag_est = batch['v_mag_est'][i]
            v_ang_est = batch['v_ang_est'][i]
            result_f = batch['result_f'][i]
            result, z_new, z_noise = case.ac_opf(active_power, reactive_power)
            COST_pre.append(result['f'])
            all_state_var = [active_power, reactive_power, v_mag_est, v_ang_est]
            action, time = agent.choose_action(z_noise, i, is_training=True) 
            next_state, reward, done = env.step(action, all_state_var)
            test_score_history.append(reward)
            actions_history.append(action)
            TIME_DRL.append(time)
            # if not i % 25:
            print(f"Testing: Reward = {reward} for Iteration # {i}/{test_size}")
        


        MAG_CHANGE = env.MAG_CHANGE
        ANG_CHANGE = env.ANG_CHANGE
        RESIDUAL_ATTACKER_DRL = env.RESIDUAL_ATTACKER_DRL
        RESIDUAL_DRL = env.RESIDUAL_DRL
        ATTACK_POSI_DRL = env.ATTACK_POSI_DRL
        TP_RANDOM_DRL = env.TP_RANDOM_DRL
        COST_after = env.COST_after
        print("ATTACK_POSI_DRL", len(ATTACK_POSI_DRL))

        print("test_score_history", test_score_history)
        # saving score values and plot
        x = [i + 1 for i in range(len(test_score_history))]
        if not os.path.exists(selected_paths['testing_score_plot_dir']):
            os.makedirs(selected_paths['testing_score_plot_dir'])
        
        if not os.path.exists(selected_paths['testing_actions_dir']):
            os.makedirs(selected_paths['testing_actions_dir'])

        actions_array = np.array(actions_history)
        csv_file_path = os.path.join(selected_paths['testing_actions_dir'], "actions_history.csv")
        np.savetxt(csv_file_path, actions_array, delimiter=",")


        agent_type = train_config['agent_type']
        test_ratio = train_config['test_ratio']
        changes_type = train_config['changes_type']
        # %% save
        # os.makedirs(os.path.join(f'simulation_data/{name}/ac/{agent_type}_ratio_{test_ratio}/{changes_type}'), exist_ok=True)
        # path = f'simulation_data/{name}/ac/{agent_type}_ratio_{test_ratio}/{changes_type}'

        os.makedirs(os.path.join(f'final_simulation_data/{name}/ac/{agent_type}_ratio_{test_ratio}/{changes_type}'), exist_ok=True)
        path = f'final_simulation_data/{name}/ac/{agent_type}_ratio_{test_ratio}/{changes_type}'


        np.save(f'{path}/{name_suffix}_TP_RANDOM_DRL.npy', TP_RANDOM_DRL/ATTACK_POSI_DRL)
        np.save(f'{path}/{name_suffix}_COST_pre.npy', COST_pre)
        np.save(f'{path}/{name_suffix}_COST_after.npy', COST_after)
        np.save(f'{path}/{name_suffix}_RESIDUAL_ATTACKER_DRL.npy', RESIDUAL_ATTACKER_DRL)

        np.save(f'{path}/{name_suffix}_RESIDUAL_DRL.npy', RESIDUAL_DRL)
        np.save(f'{path}/{name_suffix}_MAG_CHANGE.npy', MAG_CHANGE)
        np.save(f'{path}/{name_suffix}_ANG_CHANGE.npy', ANG_CHANGE)
        np.save(f'{path}/{name_suffix}_TIME_ROBUST.npy', TIME_DRL)

