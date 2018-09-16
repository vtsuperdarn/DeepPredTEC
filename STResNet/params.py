import datetime

'''
This file contains class Params for hyperparameter declarations
'''

class Params(object):
    batch_size = 64                 #defines the batch size to load in the tec data points
    map_height = 75                 # no of latitudes
    map_width = 73                  #number of longitudes
    
    tec_resolution = 5              #default value in the dateset
    data_point_freq = 2             #no of jumps for creating datapoints. If 1 at 5 mins resolution tec maps, 2 then 10 mins resolution
    
    # TEC data loading location and times
    file_dir = "../data/tec_map/filled/"
    start_date = datetime.datetime(2015, 3, 1)
    end_date = datetime.datetime(2015, 4, 1)
    
    # OMNI IMF data
    omn_dbdir = "../data/sqlite3/"
    omn_db_name = "omni_imf_res_5.sqlite"
    omn_table_name = "IMF"
    
    # train to test ratio
    train_test_ratio = 0.8

    independent_channels = True     #true if channels are independent, false if channels have shared parameter
        
    closeness_channel = True        #toggle for on/off
    closeness_freq = 2              #no of jumps for sampling. If 1 at resolution of 5 mins tec maps is taken, 2 then 10 mins resolution     
    closeness_sequence_length = 72  #no. of tec maps for closeness. if 12 then past one hourtec maps 
    
    period_channel = True           #toggle for on/off
    period_freq = 12                #no of jumps for sampling. If 12 at resolution of 1 hour tec maps is taken, 24 then 2 hours resolution 
    period_sequence_length = 72     #no. of tec maps for period. if 24 then past one day tec maps
    
    trend_channel = False           #toggle for on/off
    trend_freq = 36                 #no of jumps for sampling. If 36 at resolution of 3 hours tec maps is taken, 72 then 6 hours resolution 
    trend_sequence_length =  8      #no. of tec maps for trend. if 8 then past one day tec maps     
    
    output_freq = 2                 #no of jumps for sampling. If 1 at resolution of 5 mins tec maps is taken, 2 then 10 mins resolution 
    num_of_output_tec_maps = 24     #number of tec maps to be predicted
    resnet_out_filters = 24         #make this same as num_of_output_tec_maps
    
    num_of_filters = 12             #no. of filters in convolution layer    
    num_of_residual_units = 100     #This defines the resnet depth
    kernel_size = (5, 5)            #kernel size for convolution layer
    
    exo_values = 4                  #by, bz, vx, np
    add_exogenous = True            #True if we want to include the exogenous variable else false
    imf_normalize = True            #toggled based on the requirement in main.py and get_prediction.py
    gru_size = 32           
    gru_num_layers = 3              #incase we use stacked GRU layer
    
    #TODO initialize this
    #look_back = 
    
    load_window = 3                 #for safety so that enough data is available to create the data points
    
    epsilon = 1e-7
    beta1 = 0.9
    beta2 = 0.999
    lr = 0.0005
    delta = 0.5
    
    num_epochs = 100                  #number of epochs for training

    weight_file = "w0_mlat_45-70_1.0_mlat_80-90_1.0_mlon_None.npy"
    #weight_file = "w1_mlat_45-70_2.0_mlat_80-90_2.0_mlon_None.npy"
    weight_dir = "../WeightMatrix/"
    loss_weight_matrix = weight_dir + weight_file
    
    model_path = '../TrainedModels/model_batch'+str(batch_size)+'_epoch'+str(num_epochs)+'_resnet'+str(num_of_residual_units) +\
                 '_nresfltr'+str(resnet_out_filters) + '_nfltr'+str(num_of_filters) +\
                 '_of' + str(output_freq) + '_otec' + str(num_of_output_tec_maps) +\
                 '_cf' + str(closeness_freq) + '_csl' + str(closeness_sequence_length) +\
                 '_pf' + str(period_freq) + '_psl' + str(period_sequence_length) +\
                 '_tf' + str(trend_freq) + '_tsl' + str(trend_sequence_length) +\
                 '_gs' + str(gru_size) + '_ks' + "".join([str(x) for x in kernel_size]) +\
                 '_exo' + str(add_exogenous)[0] + '_nrm' + str(imf_normalize)[0] + \
                 '_' + weight_file[:2] + '_yr_11_13'
    
    #saved_model = '/current' # for loading the final saved model
    #saved_model = '/epoch_24' #'/epoch_X', for loading the model saved at X epoch
    saved_model = '/epoch_33' #'/epoch_X', for loading the model saved at X epoch
    
    #saved_model_path = "" #enter the model name for getting the prediction, eg. "model_batch8_epoch1_resnet10_nresfltr12_nfltr12_of1_otec12_cf1_csl12_pf12_psl24_tf36_tsl8_gs32_ks55_exoT_nrmT_1.25040221214"
    #saved_model_path = "model_batch64_epoch100_resnet50_nresfltr24_nfltr12_of2_otec24_cf2_csl48_pf12_psl72_tf36_tsl8_gs32_ks55_exoT_nrmT_yr_11_13_314.27797746658325"
    #saved_model_path = "model_batch64_epoch100_resnet100_nresfltr24_nfltr12_of2_otec24_cf2_csl48_pf12_psl72_tf36_tsl8_gs32_ks55_exoT_nrmT_yr_11_13_323.49480175971985"
    #saved_model_path = "model_batch64_epoch100_resnet100_nresfltr24_nfltr12_of2_otec24_cf2_csl72_pf12_psl72_tf36_tsl8_gs32_ks55_exoT_nrmT_yr_11_13_310.1902163028717"
    saved_model_path = "../ModelValidation/model_results/model_batch64_epoch100_resnet100_nresfltr24_nfltr12_of2_otec24_cf2_csl72_pf12_psl72_tf36_tsl8_gs32_ks55_exoT_nrmT_yr_11_13_310.1902163028717"
    #saved_model_path = "../ModelValidation/model_results/model_batch64_epoch100_resnet100_nresfltr12_nfltr12_of2_otec12_cf2_csl72_pf12_psl72_tf36_tsl8_gs32_ks55_exoT_nrmT_w0_yr_11_13_379.3419065475464"
    
    logdir = "train"
    
