import matplotlib
matplotlib.use("Agg")
from matplotlib.ticker import MultipleLocator
import datetime as dt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import dask
import os
import sys
sys.path.append("./")
from pred_err_by_hour import load_tec, calc_avg_err, add_cbar

def read_dst(file_path, stime, etime):

    # Read Dst data
    date_parser = lambda x: dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    df = pd.read_csv(file_path, parse_dates=[0], date_parser=date_parser)

    # Select datetime of interest
    df = df.loc[(df.date >= stime) & (df.date < etime), :]
    #df.set_index("date", inplace=True)

    return  df


def get_tec(pred_tec_dir, df_dst, dst_bin=[-50, -25],
            pred_time_step=10, base_model="previous_day",
            filled_tec_dir = "../data/tec_map/filled/",
            dst_filepath = "./dst-2015.csv"):

    """ Reads predicted and true TEC maps for a given Dst bin for time
    interval between stime and etime

    Parameters
    ----------
    base_model : str
        The name of the baseline models, valid inputs are "static", "previous_day"
    """

    # Select Dst in a given range
    df = df_dst.loc[(df_dst.dst >= dst_bin[0]) & (df_dst.dst < dst_bin[1]), :]


    # Generate time tags for tec maps
    dst_dtms = pd.to_datetime(df.date.values)
    dtms = [dtm + dt.timedelta(minutes=i*pred_time_step) for dtm in dst_dtms for i in range(6)]


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

    # Read the True TEC maps
    pred_tec = np.array([load_tec(x).compute() for x in pred_tec_files])
    true_tec = np.array([load_tec(x).compute() for x in true_tec_files])
    if base_tec_files is not None:
        base_tec = np.array([load_tec(x).compute() for x in base_tec_files])
    else:
        base_tec  = None

    return true_tec, pred_tec, base_tec


def err_by_dst(pred_tec_dir, stime, etime,
               mask_tec=True, mask_matrix=None, 
               pred_time_step=10,
               model_type = "STResNet",
               filled_tec_dir="../data/tec_map/filled/",
               dst_filepath = "./dst-2015.csv"):

    """ Plots three colums (relative errors for STResNet, baseline, and the ration of the two)
    for selected local times.
    """

    # Dst bins
    #dst_bins=[[-500, -100], [-100, -50], [-50, -25], [-25, 15]]
    dst_bins=[[-500, -100], [-100, -50], [-50, -20], [-20, 20]]
    dst_title = [r"Dst $<$ " + str(dst_bins[0][1]),
                 str(dst_bins[1][0]) + r"$\leq$ Dst $<$" + str(dst_bins[1][1]), 
                 str(dst_bins[2][0]) + r"$\leq$ Dst $<$" + str(dst_bins[2][1]), 
                 str(dst_bins[3][0]) + r"$\leq$ Dst $<$" + str(dst_bins[3][1])]

    # Read Dst data
    df_dst = read_dst(dst_filepath, stime, etime)

    # Create empy axes
    vmin=0; vmax=0.5; cmap="jet"
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8,8),
                             sharex=True, sharey=True)
    #fig.subplots_adjust(hspace=0.3, wspace=0.3)

    # Loop through the Dst bins
    for i, dst_bin in enumerate(dst_bins):
        ax = axes[i//2, i%2]
        true_tec, pred_tec, base_tec = get_tec(pred_tec_dir, df_dst,
                                               dst_bin=dst_bin, pred_time_step=pred_time_step,
                                               base_model="previous_day",
                                               filled_tec_dir=filled_tec_dir)
        
        if model_type == "STResNet":
            err_dict = calc_avg_err(true_tec, pred_tec)
        if model_type == "Baseline":
            err_dict = calc_avg_err(true_tec, base_tec)
        var = err_dict["Relative Average Absolute Error"]
        if mask_tec:
            var = np.ma.masked_array(var, mask_matrix, fill_value=np.nan)
        coll = ax.pcolormesh(var, cmap=cmap, vmin=vmin, vmax=vmax)

        ax.xaxis.set_major_locator(MultipleLocator(15))
        ax.yaxis.set_major_locator(MultipleLocator(15))
        #x = list(range(225, 360, 25)) + list(range(0, 35, 15)) 
        #x = list(range(325, 360, 20)) + list(range(5, 35, 20)) 
        x = list(range(-35, 0, 15)) + list(range(10, 35, 15)) 
        y = list(range(15, 95, 15))
        ax.set_xticks(range(5, 80, 15))
        ax.set_xticklabels(x, fontsize=10)
        ax.set_yticks(range(5, 80, 15))
        ax.set_yticklabels(y, fontsize=10)
        ax.set_title(dst_title[i], fontsize=10)

        #ax.set_aspect("equal")

    # add colorbar for relative error
    fig.subplots_adjust(right=0.90)
    cbar_ax = fig.add_axes([0.94, 0.30, 0.02, 0.4])
    add_cbar(fig, coll, bounds=None, cax=cbar_ax, label="Percentage")

    return fig

def main():

    # Select a model and set the path for predicted TEC map
    model = "model_batch64_epoch100_resnet100_nresfltr12_nfltr12_of2_otec12_cf2_csl72_pf12_psl72_tf36_tsl8_gs32_ks55_exoT_nrmT_w0_yr_11_13_379.3419065475464"
    model_value = model + "_values"
    model_value_dir = os.path.join("./model_results/", model_value)
    pred_tec_dir = os.path.join(model_value_dir, "predicted_tec/")
    filled_tec_dir = "../data/tec_map/filled/"

    dst_filepath = "./dst-2015.csv"


    # Error types to be plotted
    model_type = "STResNet"
    #model_type = "Baseline"

    tec_resolution = 5
    # Extract hyperparameter values from model_value folder name
    param_values = model_value.split("/")[-1].split("_")
    output_freq = [int(x.replace("of", "")) for x in param_values if x.startswith("of")][0]
    num_of_output_tec_maps = [int(x.replace("otec", "")) for x in param_values if x.startswith("otec")][0]
    pred_time_step = tec_resolution * output_freq

    stime = dt.datetime(2015, 3, 1, 1)
    etime = dt.datetime(2015, 4, 1)

    mask_tec = True
    mask_matrix = np.load("../WeightMatrix/w2_mask-2011-2013-80perc.npy")
    mask_matrix = np.logical_not(mask_matrix).astype(int)

    # Plot relative errors for STResnet and Baseline, and their ratio 
    fig = err_by_dst(pred_tec_dir, stime, etime,
                     mask_tec=mask_tec, mask_matrix=mask_matrix, 
                     pred_time_step=pred_time_step,
                     model_type=model_type,
                     filled_tec_dir=filled_tec_dir,
                     dst_filepath=dst_filepath)

    #fig_dir = "/home/muhammad/Dropbox/ARC/" + "model_otec24_exoT_w2/"
    fig_dir = "/home/muhammad/Dropbox/ARC/" + "model_otec12_exoT_w0/"
    #fig_dir = os.path.join("./plots/", model_value)

    if not os.path.exists(fig_dir):
	os.makedirs(fig_dir)
    fig_name = model_type + "_error_by_dst"
    if mask_tec:
	fig_name = fig_name + "_masked"

    fig.savefig(os.path.join(fig_dir,fig_name) + ".png", dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(fig_dir,fig_name) + ".pdf", format="pdf", bbox_inches='tight')

    plt.close(fig)

    return

if __name__ == "__main__":
    main()

