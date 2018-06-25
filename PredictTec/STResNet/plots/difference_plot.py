import datetime
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

#load the predicted tec maps
#first tec map for difference plot was at 2015-01-14 23:25
pred = np.load("predicted_tec_files/31_pred_321708.6.npy")
truth = np.load("predicted_tec_files/31_y_321708.6.npy")
tec_c = np.load("predicted_tec_files/31_close_321708.6.npy")
tec_p = np.load("predicted_tec_files/31_period_321708.6.npy")
tec_t = np.load("predicted_tec_files/31_trend_321708.6.npy")


#initializing the datetime variable
d1 = datetime.datetime(2015, 1, 15, 20, 45) 
d2 = datetime.datetime(2015, 1, 15, 20, 50)
start_date = d1
tec_resolution = (d2 - d1)
datetime_list = [d1]

#generating the datetime list for the current batch of predicted tec maps
for _ in range(1, len(pred)):
    time = datetime_list[-1]+tec_resolution
    datetime_list.append(time)
print (len(datetime_list))




#plotting the difference plot for the 32 predicted tec maps
for i in range(len(pred)):
    pred1 = np.squeeze(pred[i])
    truth1 = np.squeeze(truth[i])
    pred1[pred1 <= -0.5] = -1
    pred1[(pred1 > -0.5) & (pred1 < 0)] = 0
    difference_relative  = abs((pred1 - truth1)/truth1)
    difference_absolute  = abs(pred1 - truth1)
    
    truth1 = np.ma.masked_where(truth1 <= -1, truth1)
    pred1 = np.ma.masked_where(pred1 <= -1, pred1)
    
    fig = plt.figure(figsize=(16, 15))
    ax0 = fig.add_subplot(221)
    ax1 = fig.add_subplot(222)
    ax2 = fig.add_subplot(223)
    ax3 = fig.add_subplot(224)
    
    x = list(range(225, 360, 25)) + list(range(0, 35, 15)) 
    y = list(range(15, 90, 10))
    plt.suptitle(str(datetime_list[i]), fontsize=20)

    ax0.set_xticklabels(tuple(x), fontsize=14)
    ax0.set_yticklabels(y, fontsize=14)
    ax0.set_title('Predicted TEC', fontsize=20)


    ax1.set_xticklabels(tuple(x), fontsize=14)
    ax1.set_yticklabels(y, fontsize=14)
    ax1.set_title('True TEC', fontsize=20)


    ax2.set_xticklabels(tuple(x), fontsize=14)
    ax2.set_yticklabels(y, fontsize=14)
    ax2.set_title('Difference relative', fontsize=20)
    
    ax3.set_xticklabels(tuple(x), fontsize=14)
    ax3.set_yticklabels(y, fontsize=14)
    ax3.set_title('Difference absolute', fontsize=20)

    im0 = ax0.pcolormesh(pred1, cmap='jet', vmin=0, vmax=20)
    im1 = ax1.pcolormesh(truth1, cmap='jet', vmin=0, vmax=20)
    im2 = ax2.pcolormesh(difference_relative, cmap='jet', vmin=0, vmax=1)
    im3 = ax3.pcolormesh(difference_absolute, cmap='jet', vmin=0, vmax=20)
    
    fig.colorbar(im0, ax=ax0)
    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)
    fig.colorbar(im3, ax=ax3)
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    #plt.show()
    fig.savefig('difference_plots_both/difference31_'+str(i+1)+'.png', dpi=fig.dpi, bbox_inches='tight')
