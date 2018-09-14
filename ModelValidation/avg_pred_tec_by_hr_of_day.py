import datetime as dt
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from params import Params as param
import os

model_value_dir = "model_batch64_epoch100_resnet100_nresfltr24_nfltr12_of2_otec24_cf2_csl72_pf12_psl72_tf36_tsl8_gs32_ks55_exoT_nrmT_yr_11_13_310.1902163028717_values"
pred_tec_dir = os.path.joing(model_value_dir, "/predicted_tec/")
pred_tec_path = os.path.joing("/sd-data/DeepPredTEC/ModelValidation/", pred_tec_dir)

stime = dt.datetime(2015, 3, 1)
etime = dt.datetime(2015, 4, 1)

#getting the datetime variables for the predicted TEC maps
tec_resolution = dt.timedelta(minutes=param.tec_resolution)
datetime_list = [stime]
for _ in range(1, param.num_of_output_tec_maps):
    time = datetime_list[-1]+tec_resolution
    datetime_list.append(time)

#reading the numpy arrays
path = model_path+'/'+folder_name+'/'
pred = np.load(path+"pred.npy")
truth = np.load(path+"y.npy")
tec_c = np.load(path+"close.npy")
tec_p = np.load(path+"period.npy")
tec_t = np.load(path+"trend.npy")

#transpose for iterating over the TEC maps
pred = np.transpose(pred, [2, 0, 1])
truth = np.transpose(truth, [2, 0, 1])
tec_c = np.transpose(tec_c, [2, 0, 1])
tec_p = np.transpose(tec_p, [2, 0, 1])
tec_t = np.transpose(tec_t, [2, 0, 1])


for i in range(param.num_of_output_tec_maps):
    #(1, 75, 73) -> (75, 73). Making it 2D for plotting
    pred1 = np.squeeze(pred[i])
    truth1 = np.squeeze(truth[i])
    tec_c1 = np.squeeze(tec_c[i])
    tec_p1 = np.squeeze(tec_p[i])
    tec_t1 = np.squeeze(tec_t[i])    
    
    fig = plt.figure(figsize=(23, 13))
    x = list(range(225, 360, 25)) + list(range(0, 35, 15)) 
    y = list(range(15, 90, 10))

    #for neglecting the missing values in the plot
    truth1 = np.ma.masked_where(truth1 <= -1, truth1)
    #setting it -0.5 (heuristic) for predicted maps
    pred1 = np.ma.masked_where(pred1 <= -0.5, pred1)
    
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(233)
    ax3 = fig.add_subplot(234)
    ax4 = fig.add_subplot(235)
    ax5 = fig.add_subplot(236)

    plt.suptitle(str(datetime_list[i]), fontsize=20)

    ax1.set_xticklabels(tuple(x), fontsize=14)
    ax1.set_yticklabels(y, fontsize=14)
    ax1.set_title('Predicted TEC', fontsize=20)

    ax2.set_xticklabels(tuple(x), fontsize=14)
    ax2.set_yticklabels(y, fontsize=14)
    ax2.set_title('True TEC', fontsize=20)

    ax3.set_xticklabels(tuple(x), fontsize=14)
    ax3.set_yticklabels(y, fontsize=14)
    ax3.set_title('Channel Closeness', fontsize=20)

    ax4.set_xticklabels(tuple(x), fontsize=14)
    ax4.set_yticklabels(y, fontsize=14)
    ax4.set_title('Channel Period', fontsize=20)

    ax5.set_xticklabels(tuple(x), fontsize=14)
    ax5.set_yticklabels(y, fontsize=14)
    ax5.set_title('Channel Trend', fontsize=20)

    a = -30
    b= 20
    im1 = ax1.pcolormesh(pred1, cmap='jet', vmin=0, vmax=20)
    im2 = ax2.pcolormesh(truth1, cmap='jet', vmin=0, vmax=20)
    im3 = ax3.pcolormesh(tec_c1, cmap='jet', vmin=a, vmax=b)
    im4 = ax4.pcolormesh(tec_p1, cmap='jet', vmin=a, vmax=b)
    im5 = ax5.pcolormesh(tec_t1, cmap='jet', vmin=a, vmax=b)

    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)
    fig.colorbar(im3, ax=ax3)
    fig.colorbar(im4, ax=ax4)
    fig.colorbar(im5, ax=ax5)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.savefig(path+str(i+1)+'.png', dpi=fig.dpi, bbox_inches='tight')
