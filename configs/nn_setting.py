"""
Contains the configuration of neural network settings and agent parameters.
"""

# Neural network settings
nn_config = {
    'Act_fc1_dims': 128,
    'Act_fc2_dims': 128,
    'Act_fc3_dims': 64,
    'Act_fc4_dims': 32,
    'Crtc_fc1_dims': 128,
    'Crtc_fc2_dims': 128,
    'Crtc_fc3_dims': 256,
    'Crtc_fc4_dims': 128,
}

# Agent settings
agent_config = {
    # try these 4 different LRs
    # 'alpha':0.0001, 'beta':0.0001,  # -->1------(worst)
    # 'alpha':0.0001, 'beta':0.001,  # -->2-----(worst)
    # 'alpha':0.001, 'beta':0.0001,  # -->3
    'alpha':0.001, 'beta':0.001,  # -->4
    # 'alpha':0.0005, 'beta':0.001,  # -->5----------keep (best)
    # 'alpha':0.002, 'beta':0.001,  # -->6
    # 'alpha':0.001, 'beta':0.0005,  # -->7----------keep (better)
    # 'alpha':0.001, 'beta':0.002,  # -->8
    # 'alpha':0.0005, 'beta':0.0005,  # -->5----------keep dr
    'tau': 0.001,
    'batch_size': 128,  
}

