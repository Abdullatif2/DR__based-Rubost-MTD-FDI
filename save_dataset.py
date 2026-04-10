
# %%
from utils.initialization import *
from utils.mtd_incomplete import run_incomplete
from utils.mtd_complete import run_complete
from utils.utils import x_to_b, find_posi
from copy import deepcopy
from utils.grid_fun import dc_grid
from numpy.linalg import norm
import matplotlib.pyplot as plt
from utils.evaluation_fun import generate_data
import os
from torch.profiler import profile
print(name)
print(name_suffix)

import pandas as pd

# Initialize lists to hold multiple samples
active_power_list = []
reactive_power_list = []
v_mag_est_list = []
v_ang_est_list = []
Jr_N_list = []
result_f_list = []
#%
# training dataset
np.random.seed(2)
for i in range(30000):  
    active_power, reactive_power, v_mag_est, v_ang_est, Jr_N, result = generate_data(case)
    print("loop iter ", i)

    active_power_list.append(active_power.tolist())
    reactive_power_list.append(reactive_power.tolist())
    v_mag_est_list.append(v_mag_est.tolist())
    v_ang_est_list.append(v_ang_est.tolist())
    # Jr_N_list.append(Jr_N.tolist())
    result_f_list.append(result['f']) 


df = pd.DataFrame({
    'active_power': [str(ap) for ap in active_power_list],
    'reactive_power': [str(ap) for ap in reactive_power_list],
    'v_mag_est': [str(ap) for ap in v_mag_est_list],  
    'v_ang_est': [str(ap) for ap in v_ang_est_list],  
    # 'Jr_N': [str(ap) for ap in Jr_N_list],  
    'result_f': result_f_list
})

# df.to_csv('dataset_case57_training.csv', index=False)
df.to_csv('dataset_case57_training_min.csv', index=False)




# #%%
# import pandas as pd
# import numpy as np


# # Example loop where generate_data(case) is called multiple times
# for i in range(50):  # Assuming 'cases' is the iterable you're looping through
#     active_power, reactive_power, v_mag_est, v_ang_est, Jr_N, result = generate_data(case)
    
#     # Append the new sample to each list
#     active_power_list.append(active_power)
#     reactive_power_list.append(reactive_power)
#     v_mag_est_list.append(v_mag_est)
#     v_ang_est_list.append(v_ang_est)
#     Jr_N_list.append(Jr_N)
#     result_f_list.append(result['f'])  # Assuming result['f'] is what you're interested in




# df = pd.DataFrame({
#     'active_power_1': [ap[0] for ap in active_power],
#     'active_power_2': [ap[1] for ap in active_power],
#     # Continue for the rest of the elements in active_power
#     'reactive_power_1': [rp[0] for rp in reactive_power],
#     'reactive_power_2': [rp[1] for rp in reactive_power],
#     # Continue for the rest of the elements in reactive_power
#     'v_mag_est_1': [vm[0] for vm in v_mag_est],
#     'v_ang_est_1': [va[0] for va in v_ang_est],
#     'Jr_N_1': [jr[0] for jr in Jr_N],
#     # Add more columns depending on the number of elements in each array
#     'result_f': result_f
# })



# # Save the DataFrame to a CSV file
# df.to_csv('dataset.csv', index=False)













# # #%% reading 

# import pandas as pd

# # Read the CSV file
# df = pd.read_csv('dataset_case14_training.csv')

# # Extract each column into separate variables
# active_power = df['active_power'].values
# reactive_power = df['reactive_power'].values
# v_mag_est = df['v_mag_est'].values
# v_ang_est = df['v_ang_est'].values
# # Jr_N = df['Jr_N'].values
# result_f = df['result_f'].values  # This corresponds to 'result['f']'

# # Optionally, you can reconstruct the result dictionary if needed
# result = [{'f': f_value} for f_value in result_f]
# print(len(result))
# # Now you have active_power, reactive_power, v_mag_est, v_ang_est, Jr_N, and result

# print("active_power")


# # #%% reading 

# # import pandas as pd
# # import ast
# # # Read the CSV file
# # df = pd.read_csv('dataset.csv')

# # active_power = df['active_power'].apply(lambda x: ast.literal_eval(x.strip())).values
# # reactive_power = df['reactive_power'].apply(lambda x: ast.literal_eval(x.strip())).values
# # v_mag_est = df['v_mag_est'].apply(lambda x: ast.literal_eval(x.strip())).values
# # v_ang_est = df['v_ang_est'].apply(lambda x: ast.literal_eval(x.strip())).values
# # # Jr_N = df['Jr_N'].apply(lambda x: ast.literal_eval(x.strip())).values
# # result_f = df['result_f'].values



