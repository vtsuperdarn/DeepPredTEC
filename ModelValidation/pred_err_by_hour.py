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

def get_tec(stime, etime, pred_tec_dir, time_window=[0, 1],
            pred_time_step=10, base_model="previous_day",
            filled_tec_dir = "../data/tec_map/filled/"):

    """ Reads predicted and true TEC maps for a given time_windown for time 
    interval between stime and etime

    Parameters
    ----------
    base_model : str
        The name of the baseline models, valid inputs are "static", "previous_day"
    """

    # Set the stime minute to 0
    stime = stime.replace(minute=0)

    # Generate time tags for tec maps
    num_dtms = int((etime-stime).total_seconds() / 60. / 10.)
    dtms = [stime + dt.timedelta(minutes=i*pred_time_step) for i in range(num_dtms)]

    # Select time tags within time_window
    dtms = [x for x in dtms if (((60*x.hour+x.minute)>60*time_window[0]) &\
                                ((60*x.hour+x.minute)<=60*time_window[1]))]
    
    # Get corresponding file names
    pred_tec_files = [os.path.join(pred_tec_dir, x.strftime("%Y%m%d.%H%M") + "_pred.npy")\
                      for x in dtms]
    true_tec_files = [os.path.join(filled_tec_dir, x.strftime("%Y%m%d"),\
                                   x.strftime("%Y%m%d.%H%M") + ".npy") for x in dtms]

    if base_model is None:
        base_tec_files = None
    elif base_model == "previous_day":
        base_tec_files = [os.path.join(filled_tec_dir, (x-dt.timedelta(days=1)).strftime("%Y%m%d"),\
                         (x-dt.timedelta(days=1)).strftime("%Y%m%d.%H%M") + ".npy") for x in dtms]
    elif base_model == "static":
        base_tec_files = None

    # Read the True TEC maps
    pred_tec = np.array([load_tec(x).compute() for x in pred_tec_files])
    true_tec = np.array([load_tec(x).compute() for x in true_tec_files])
    if base_tec_files is not None:
        base_tec = np.array([load_tec(x).compute() for x in base_tec_files])
    else:
        base_tec  = None

    return true_tec, pred_tec, base_tec

def calc_avg_err(true_tec, pred_tec):
    """ Calculates the average predicted TEC error
    """

    err_dict = {}
    pred_true_diff =  np.abs(pred_tec - true_tec)
    abs_avg_err = pred_true_diff.mean(axis=0)
    err_dict["pred_true_diff"] = pred_true_diff
    err_dict["True Average"] = true_tec.mean(axis=0)
    err_dict["Predicted Average"] = pred_tec.mean(axis=0)
    err_dict["Average Absolute Error"] = abs_avg_err
    err_dict["Average Absolute Error Std"] = pred_true_diff.std(axis=0)
    err_dict["abs_avg_err_max"] = pred_true_diff.max(axis=0)
    err_dict["abs_avg_err_min"] = pred_true_diff.min(axis=0)
    err_dict["Relative Average Absolute Error"] = np.divide(abs_avg_err,
                                                            true_tec.mean(axis=0))

    return err_dict

def add_cbar(fig, coll, bounds=None, label="TEC Unit", cax=None):

    from matplotlib.ticker import MultipleLocator
    # add color bar
    if cax:
        cbar=fig.colorbar(coll, cax=cax, orientation="vertical",
                          boundaries=bounds, drawedges=False)
    else:
        cbar=fig.colorbar(coll, orientation="vertical", shrink=.65,
                          boundaries=bounds, drawedges=False)

    #define the colorbar labels
    if bounds:
        l = []
        for i in range(0,len(bounds)):
            if i == 0 or i == len(bounds)-1:
                l.append(' ')
                continue
            l.append(str(int(bounds[i])))
        cbar.ax.set_yticklabels(l)
    else:
        for i in [0, -1]:
            lbl = cbar.ax.yaxis.get_ticklabels()[i]
            lbl.set_visible(False)
    #cbar.ax.tick_params(axis='y',direction='out')
    cbar.set_label(label, fontsize=12)

    return

def main():

    # Select a model and set the path for predicted TEC map
    #model_value = "model_batch64_epoch100_resnet100_nresfltr24_nfltr12_of2_otec24_cf2_csl72_pf12_psl72_tf36_tsl8_gs32_ks55_exoT_nrmT_yr_11_13_310.1902163028717_values"
    model_value = "model_batch64_epoch100_resnet100_nresfltr12_nfltr12_of2_otec12_cf2_csl72_pf12_psl72_tf36_tsl8_gs32_ks55_exoT_nrmT_w0_yr_11_13_379.3419065475464_values"
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
    base_model = "previous_day"

#    model = "STResNet"
#    err_types = ["Relative Average Absolute Error",
#		 "Average Absolute Error", "Average Absolute Error Std",
#		 "True Average", "Predicted Average"]

    model = "Baseline"
    err_types = ["Relative Average Absolute Error",
		 "Average Absolute Error", "Average Absolute Error Std"]

    window_len = 1 # Hour
    window_dist = 1 # Hour, skips every window_dist hours
    start_hours = [x for x in range(0, 24-window_len, window_len+window_dist)]
    end_hours = [x for x in range(window_len, 24, window_len+window_dist)]

    for err_type in err_types:
        if err_type in ["Average Absolute Error"]:
            vmin=0; vmax=10; cbar_label="TEC Unit"
        if err_type in ["Relative Average Absolute Error"]:
            vmin=0; vmax=0.5; cbar_label="Ratio"
        if err_type in ["Average Absolute Error Std"]:
            vmin=0; vmax=5; cbar_label="TEC Unit"
        if err_type in ["True Average", "Predicted Average"]:
            vmin=0; vmax=20; cbar_label="TEC Unit"

	# Create empy axes
	fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12,10),
				 sharex=True, sharey=True)
	fig.subplots_adjust(hspace=0.3, wspace=0.3)
	if len(axes) > 1:
	    axes = [ax for lst in axes for ax in lst]
	else:
	    axes = [axes]
	
	# Loop through the time windows
	for i in range(len(start_hours)): 
	    ax = axes[i]
	    time_window = [start_hours[i], end_hours[i]]
	    true_tec, pred_tec, base_tec = get_tec(stime, etime, pred_tec_dir,
						   time_window=time_window,
						   pred_time_step=pred_time_step,
						   base_model=base_model,
						   filled_tec_dir=filled_tec_dir)
	    
	    if model == "STResNet":
		err_dict = calc_avg_err(true_tec, pred_tec)
	    if model == "Baseline":
		err_dict = calc_avg_err(true_tec, base_tec)

	    if base_tec is not None:
		err_dict_base = calc_avg_err(true_tec, pred_tec)

	    coll = ax.pcolormesh(err_dict[err_type], cmap='jet', vmin=vmin, vmax=vmax)
	    #x = list(range(225, 360, 25)) + list(range(0, 35, 15)) 
	    #x = list(range(325, 360, 20)) + list(range(5, 35, 20)) 
	    x = list(range(-35, 0, 20)) + list(range(5, 35, 20)) 
	    y = list(range(15, 90, 20))
	    ax.set_xticklabels(tuple(x), fontsize=10)
	    ax.set_yticklabels(y, fontsize=10)
	    ax.set_title("UT Hour = " + str(time_window))

	# Set the Super Title
        fig_title = model + " Model, " + err_type + " for " +\
		    stime.strftime("%b %d, %Y") + " - " + etime.strftime("%b %d, %Y")
	plt.suptitle(fig_title, x=0.5, y=0.95, fontsize=12)

	# add colorbar
	fig.subplots_adjust(right=0.90)
	cbar_ax = fig.add_axes([0.93, 0.25, 0.02, 0.5])
	add_cbar(fig, coll, bounds=None, cax=cbar_ax, label=cbar_label)

	fig_dir = "/home/muhammad/Dropbox/ARC/" + "model_2/"
	fig_name = model + "_" + "_".join(err_type.split()) + ".png"
	fig.savefig(fig_dir+fig_name, dpi=200, bbox_inches='tight')

        plt.close(fig)

    return

if __name__ == "__main__":
    main()

