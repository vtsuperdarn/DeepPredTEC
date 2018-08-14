'''
This file contains class Params for hyperparameter declarations
'''

class Params(object):
    batch_size = 32         #defines the batch size to load in the tec data points
    map_height = 75         # no of latitudes
    map_width = 73          #number of longitudes
    tec_resolution = 5
    closeness_freq = 1      #no of jumps for sampling. If 1 at resolution of 5 mins tec maps is taken, 2 then 10 mins resolution     
    period_freq = 12        #no of jumps for sampling. If 12 at resolution of 1 hour tec maps is taken, 24 then 2 hours resolution 
    trend_freq = 36         #no of jumps for sampling. If 36 at resolution of 3 hours tec maps is taken, 72 then 6 hours resolution 
    output_freq = 1         #no of jumps for sampling. If 1 at resolution of 5 mins tec maps is taken, 2 then 10 mins resolution 
    closeness_sequence_length = 12  #no. of tec maps for closeness. if 12 then past one hourtec maps 
    period_sequence_length = 24     #no. of tec maps for period. if 24 then past one day tec maps 
    trend_sequence_length =  8      #no. of tec maps for trend. if 8 then past one day tec maps     
    num_of_filters = 12             #no. of filters in convolution layer    
    num_of_residual_units = 10      #This defines the resnet depth
    num_of_output_tec_maps = 12     #number of tec maps to be predicted
    resnet_out_filters = 12         #make this same as num_of_output_tec_maps
    imf_normalize = True            #toggled based on the requirement in main.py and get_prediction.py
    add_exogenous = True            #True if we want to include the exogenous variable else false
    exo_values = 4                  #by, bz, vx, np
    gru_size = 32           
    gru_num_layers = 3              #incase we use stacked GRU layer
    load_window = 1                 #for safety so that enough data is available to create the data points
    kernel_size = (5, 5)            #kernel size for convolution layer
    epsilon = 1e-7
    beta1 = 0.8
    beta2 = 0.999
    lr = 0.0005
    num_epochs = 1                  #number of epochs for training
    delta = 0.5
    model_path = './model_batch'+str(batch_size)+'_epoch'+str(num_epochs)+'_resnet'+str(num_of_residual_units) +\
                 '_nresfltr'+str(resnet_out_filters) + '_nfltr'+str(num_of_filters) +\
                 '_of' + str(output_freq) + '_otec' + str(num_of_output_tec_maps) +\
                 '_cf' + str(closeness_freq) + '_csl' + str(closeness_sequence_length) +\
                 '_pf' + str(period_freq) + '_psl' + str(period_sequence_length) +\
                 '_tf' + str(trend_freq) + '_tsl' + str(trend_sequence_length) +\
                 '_gs' + str(gru_size) + '_ks' + "".join([str(x) for x in kernel_size]) +\
                 '_exo' + str(add_exogenous)[0] + '_nrm' + str(imf_normalize)[0]
    saved_model = '/current' # for loading the final saved model
    #saved_model = '/epoch_0' #'/epoch_X', for loading the model saved at X epoch
    saved_model_path = "" #enter the model name for getting the prediction, eg. "model_batch8_epoch1_resnet10_nresfltr12_nfltr12_of1_otec12_cf1_csl12_pf12_psl24_tf36_tsl8_gs32_ks55_exoT_nrmT_1.25040221214"
    logdir = "train"
