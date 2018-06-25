import datetime
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


#load the predicted tec maps
pred = np.load("predicted_tec_files/30_pred_288137.0.npy")
truth = np.load("predicted_tec_files/30_y_288137.0.npy")
tec_c = np.load("predicted_tec_files/30_close_288137.0.npy")
tec_p = np.load("predicted_tec_files/30_period_288137.0.npy")
tec_t = np.load("predicted_tec_files/30_trend_288137.0.npy")



#converting the tec maps into the required shape of (75, 73, 1)
pred = np.squeeze(pred)
tec_c = np.squeeze(tec_c)
tec_p = np.squeeze(tec_p)
tec_t = np.squeeze(tec_t)
pred = np.expand_dims(pred, axis=3)
tec_c = np.expand_dims(tec_c, axis=3)
tec_p = np.expand_dims(tec_p, axis=3)
tec_t = np.expand_dims(tec_t, axis=3)
print pred.shape
print tec_c.shape
print tec_p.shape
print tec_t.shape
print truth.shape




#initializing the datetime variable
d1 = datetime.datetime(2015, 1, 15, 18, 5) 
d2 = datetime.datetime(2015, 1, 15, 18, 10)
start_date = d1
tec_resolution = (d2 - d1)
datetime_list = [d1]

#generating the datetime list for the current batch of predicted tec maps
for _ in range(1, len(pred)):
    time = datetime_list[-1]+tec_resolution
    datetime_list.append(time)
print (len(datetime_list))



#plotting the channel wise plots for the 32 (batch size) predicted tec maps
for i in range(len(pred)):
    pred1 = np.squeeze(pred[i])
    truth1 = np.squeeze(truth[i])
    tec_c1 = np.squeeze(tec_c[i])
    tec_p1 = np.squeeze(tec_p[i])
    tec_t1 = np.squeeze(tec_t[i])
    
    fig = plt.figure(figsize=(23, 13))
    x = list(range(225, 360, 25)) + list(range(0, 35, 15)) 
    y = list(range(15, 90, 10))

    truth1 = np.ma.masked_where(truth1 <= -1, truth1)
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
    b = 20
    im1 = ax1.pcolormesh(pred1, cmap='jet', vmin=0, vmax=20)
    im2 = ax2.pcolormesh(truth1, cmap='jet', vmin=0, vmax=20)
    im3 = ax3.pcolormesh(tec_c1, cmap='jet', vmin=a, vmax=b)
    im2 = ax4.pcolormesh(tec_p1, cmap='jet', vmin=a, vmax=b)
    im3 = ax5.pcolormesh(tec_t1, cmap='jet', vmin=a, vmax=b)

    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)
    fig.colorbar(im3, ax=ax3)
    fig.colorbar(im2, ax=ax4)
    fig.colorbar(im3, ax=ax5)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    #plt.show()
    fig.savefig('channel_wise/30_'+str(i+1)+'.png', dpi=fig.dpi, bbox_inches='tight')
