import matplotlib
matplotlib.use("Agg")
import datetime as dt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import dask
import os
import sys
sys.path.append("./")
from pred_err_by_hour import get_tec, calc_avg_err

@dask.delayed
def load_tec(file_path):
    x = np.load(file_path)
    return x

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
            #if i == 0 or i == len(bounds)-1:
            if i == len(bounds)-1:
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

def relative_err_plot(pred_tec_dir, stime, etime,
 		      mask_tec=True, mask_matrix=None, 
		      pred_time_step=10,
		      filled_tec_dir="../data/tec_map/filled/"):
    """ Plots three colums (relative errors for STResNet, baseline, and the ration of the two)
    for selected local times.
    """

    # Error types to be plotted
    err_types = ["STResNet Relative Error", "Baseline Relative Error",
                 "Relative Error Ratio"]

    # Selected Hours
    window_len = 1 # Hour
    window_dist = 3 # Hour, skips every window_dist hours
    start_hours = [x for x in range(0, 24-window_len, window_len+window_dist)]
    end_hours = [x for x in range(window_len, 24, window_len+window_dist)]

    # Create empy axes
    fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(12,6),
                             sharex=True, sharey=True)
    #fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for j, err_type in enumerate(err_types):
        if err_type in ["STResNet Relative Error", "Baseline Relative Error"]:
            vmin=0; vmax=0.5; cmap="jet"
        if err_type in ["Relative Error Ratio"]:
            vmin=1; vmax=3; cmap="Reds"
	
	# Loop through the time windows
	for i in range(len(start_hours)): 
	    ax = axes[j, i]
	    time_window = [start_hours[i], end_hours[i]]
	    true_tec, pred_tec, base_tec = get_tec(stime, etime, pred_tec_dir,
						   time_window=time_window,
						   pred_time_step=pred_time_step,
						   base_model="previous_day",
						   filled_tec_dir=filled_tec_dir)
	    
	    err_dict_stresnet = calc_avg_err(true_tec, pred_tec)
	    err_dict_baseline = calc_avg_err(true_tec, base_tec)
	    err_dict_baseline["Relative Error Ratio"] = np.divide(err_dict_baseline["Relative Average Absolute Error"],
								  err_dict_stresnet["Relative Average Absolute Error"])

	    if err_type == "STResNet Relative Error":
		var = err_dict_stresnet["Relative Average Absolute Error"]
	    if err_type == "Baseline Relative Error":
		var = err_dict_baseline["Relative Average Absolute Error"]
	    if err_type == "Relative Error Ratio":
		var = err_dict_baseline["Relative Error Ratio"]
            if mask_tec:
                var = np.ma.masked_array(var, mask_matrix, fill_value=np.nan)
	    coll = ax.pcolormesh(var, cmap=cmap, vmin=vmin, vmax=vmax)
	    if err_type in ["STResNet Relative Error", "Baseline Relative Error"]:
		coll_percent = coll
	    if err_type in ["Relative Error Ratio"]:
		coll_ratio = coll

	    #x = list(range(225, 360, 25)) + list(range(0, 35, 15)) 
	    #x = list(range(325, 360, 20)) + list(range(5, 35, 20)) 
	    x = list(range(-35, 0, 20)) + list(range(5, 35, 20)) 
	    y = list(range(15, 90, 20))
	    ax.set_xticklabels(tuple(x), fontsize=10)
	    ax.set_yticklabels(y, fontsize=10)
	    if j == 0:
		ax.set_title("UT Hour = " + str(time_window), fontsize=10)

    # add colorbar for relative error
    fig.subplots_adjust(right=0.92)
    cbar_ax1 = fig.add_axes([0.94, 0.48, 0.015, 0.3])
    add_cbar(fig, coll_percent, bounds=None, cax=cbar_ax1, label="Percentage")

    # add colorbar for relative error ratio
    cbar_ax2 = fig.add_axes([0.94, 0.12, 0.015, 0.2])
    add_cbar(fig, coll_ratio, bounds=None, cax=cbar_ax2, label="Ratio")

    # Set the Super Title
    axes[0,0].annotate("STResNet", xy=(-0.4,0.5), va="center", xycoords="axes fraction", fontsize=12, rotation=90)
    axes[1,0].annotate("Baseline", xy=(-0.4,0.5), va="center", xycoords="axes fraction", fontsize=12, rotation=90)
    axes[2,0].annotate("Baseline/STResNet", xy=(-0.4,0.5), va="center", xycoords="axes fraction", fontsize=10, rotation=90)

    return fig

def main():

    # Select a model and set the path for predicted TEC map
    model = "model_batch64_epoch100_resnet100_nresfltr12_nfltr12_of2_otec12_cf2_csl72_pf12_psl72_tf36_tsl8_gs32_ks55_exoT_nrmT_w0_yr_11_13_379.3419065475464"
    model_value = model + "_values"
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

    mask_tec = True
    mask_matrix = np.load("../WeightMatrix/w2_mask-2011-2013-80perc.npy")
    mask_matrix = np.logical_not(mask_matrix).astype(int)

    # Plot relative errors for STResnet and Baseline, and their ratio 
    fig = relative_err_plot(pred_tec_dir, stime, etime,
			    mask_tec=mask_tec, mask_matrix=mask_matrix, 
			    pred_time_step=pred_time_step,
			    filled_tec_dir=filled_tec_dir)

    #fig_dir = "/home/muhammad/Dropbox/ARC/" + "model_otec24_exoT_w2/"
    fig_dir = "/home/muhammad/Dropbox/ARC/" + "model_otec12_exoT_w0/"
    #fig_dir = os.path.join("./plots/", model_value)

    if not os.path.exists(fig_dir):
	os.makedirs(fig_dir)
    fig_name = "relative_error_plot"
    if mask_tec:
	fig_name = fig_name + "_masked"

    fig.savefig(os.path.join(fig_dir,fig_name) + ".png", dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(fig_dir,fig_name) + ".pdf", format="pdf", bbox_inches='tight')

    plt.close(fig)

    return

if __name__ == "__main__":
    main()

