'''
Author: Sneha Singhania
Date: June 3, 2018
Comment: This file contains class Params for hyperparameter declarations
'''

class Params(object):
    batch_size = 32
    map_height = 75
    map_width = 73
    closeness_sequence_length = 12
    period_sequence_length = 24
    trend_sequence_length =  8
    num_of_filters = 64
    num_of_residual_units = 5 #increase this value during the actual training
    num_of_output_tec_maps = 1
    look_back = 288
    exo_values = 4 #by, bz, vx, np
    lstm_size = 64
    num_layers = 3
    epsilon = 1e-7
    beta1 = 0.8
    beta2 = 0.999
    lr = 0.001
    num_epochs = 20
    delta = 0.5
    model_path = './exo_model'
    

