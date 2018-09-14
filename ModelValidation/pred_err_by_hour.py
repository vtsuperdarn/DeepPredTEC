import matplotlib
matplotlib.use("Agg")
import datetime as dt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import dask
import os

@dask.delayed
def load_tec(file_path):
    x = np.load(file_path)
    return x

def get_tec(stime, etime, pred_tec_dir, time_window=[0, 1], pred_time_step=10,
            filled_tec_dir = "../data/tec_map/filled/"):

    """ Reads predicted and true TEC maps for a given time_windown for time 
    interval between stime and etime"""

    # Set the stime minute to 0
    stime = stime.replace(minute=0)

    # Generate time tags for tec maps
    num_dtms = int((etime-stime).total_seconds() / 60. / 10.)
    dtms = [stime + dt.timedelta(minutes=i*pred_time_step) for i in range(num_dtms)]

    # Select time tags within time_window
    dtms = [x for x in dtms if ((x.hour>=time_window[0]) & (x.hour<time_window[1]))]
    
    # Get corresponding file names
    pred_tec_files = [os.path.join(pred_tec_dir, x.strftime("%Y%m%d.%H%M") + "_pred.npy")\
                      for x in dtms]
    true_tec_files = [os.path.join(filled_tec_dir, x.strftime("%Y%m%d.%H%M") + ".npy")\
                      for x in dtms]

    # Read the True TEC maps
    pred_tec = [load_tec(x).compute() for x in pred_tec_files]


    return true_tec, pred_tec

def calc_avg_tec(stime, etime, time_window, pred_tec_dir, error_type="absolute",
                 base_model="static", pred_time_step=10,
                 filled_tec_dir = "../data/tec_map/filled/"):
    """ Calculates the average predicted TEC error for each time_window for 
    the time interval between stime and etime
    
    Parameters
    ----------
    base_model : str
        The name of the baseline models, valid inputs are "static", "previous_day"
    error_type : str
        The type of error values

    """



    if error_type == "absolute":
        pass

    elif error_type == "relative":
        pass

    else:
        avg_tec_err = 0

    # update the current time



    return avg_tec_err





def main():

    # Select a model and set the path for predicted TEC map
    model_value = "model_batch64_epoch100_resnet100_nresfltr24_nfltr12_of2_otec24_cf2_csl72_pf12_psl72_tf36_tsl8_gs32_ks55_exoT_nrmT_yr_11_13_310.1902163028717_values"
    model_value_dir = os.path.join("./model_results/", model_value)
    pred_tec_dir = os.path.join(model_value_dir, "predicted_tec/")
    filled_tec_dir = "../data/tec_map/filled/"

    tec_resolution = 5
    # Extract hyperparameter values from model_value folder name
    param_values = model_value.split("/")[-1].split("_")
    output_freq = [int(x.replace("of", "")) for x in param_values if x.startswith("of")][0]
    num_of_output_tec_maps = [int(x.replace("otec", "")) for x in param_values if x.startswith("otec")][0]

    pred_time_step = tec_resolution * output_freq

    stime = dt.datetime(2015, 3, 1)
    etime = dt.datetime(2015, 4, 1)

    window_len = 1 # Hour
    window_dist = 1 # Hour, skips every window_dist hours
    start_hours = [x for x in range(0, 24-window_len, window_len+window_dist)]
    end_hours = [x for x in range(window_len, 24, window_len+window_dist)]
    
    N = 1

    # Loop through the time windows
    for i in range(len(start_hours)): 
        time_window = [start_hours[i], end_hours[i]]
        aa = get_tec(stime, etime, pred_tec_dir, time_window=time_window,
                     pred_time_step=pred_time_step, 
                     filled_tec_dir=filled_tec_dir)
        #avg_tec_err = calc_avg_tec()
        N = 1


#    #getting the datetime variables for the predicted TEC maps
#    tec_resolution = dt.timedelta(minutes=tec_resolution)
#    datetime_list = [stime]
#    for _ in range(1, num_of_output_tec_maps):
#        time = datetime_list[-1]+tec_resolution
#        datetime_list.append(time)
#
#    #reading the numpy arrays
#    pred = np.load(pred_tec_path+"_pred.npy")
#    truth = np.load(pred_tec_path+"_true.npy")
#
#    #transpose for iterating over the TEC maps
#    pred = np.transpose(pred, [2, 0, 1])
#    truth = np.transpose(truth, [2, 0, 1])
#
#    for i in range(num_of_output_tec_maps):
#        #(1, 75, 73) -> (75, 73). Making it 2D for plotting
#        pred1 = np.squeeze(pred[i])
#        truth1 = np.squeeze(truth[i])
#        
#        fig = plt.figure(figsize=(23, 13))
#        x = list(range(225, 360, 25)) + list(range(0, 35, 15)) 
#        y = list(range(15, 90, 10))
#
#        #for neglecting the missing values in the plot
#        truth1 = np.ma.masked_where(truth1 <= -1, truth1)
#        #setting it -0.5 (heuristic) for predicted maps
#        pred1 = np.ma.masked_where(pred1 <= -0.5, pred1)
#        
#        ax1 = fig.add_subplot(231)
#        ax2 = fig.add_subplot(233)
#        ax3 = fig.add_subplot(234)
#        ax4 = fig.add_subplot(235)
#        ax5 = fig.add_subplot(236)
#
#        plt.suptitle(str(datetime_list[i]), fontsize=20)
#
#        ax1.set_xticklabels(tuple(x), fontsize=14)
#        ax1.set_yticklabels(y, fontsize=14)
#        ax1.set_title('Predicted TEC', fontsize=20)
#
#        ax2.set_xticklabels(tuple(x), fontsize=14)
#        ax2.set_yticklabels(y, fontsize=14)
#        ax2.set_title('True TEC', fontsize=20)
#
#        a = -30
#        b= 20
#        im1 = ax1.pcolormesh(pred1, cmap='jet', vmin=0, vmax=20)
#        im2 = ax2.pcolormesh(truth1, cmap='jet', vmin=0, vmax=20)
#
#        fig.colorbar(im1, ax=ax1)
#        fig.colorbar(im2, ax=ax2)
#        fig.tight_layout()
#        fig.subplots_adjust(top=0.95)
#        fig.savefig(model_value_dir+str(i+1)+'.png', dpi=fig.dpi, bbox_inches='tight')
    return

if __name__ == "__main__":
    main()

