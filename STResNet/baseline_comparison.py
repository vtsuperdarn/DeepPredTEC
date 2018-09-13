import datetime
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from params import Params as param


print ("Enter the model_path_values name:")
model_path = raw_input()

print ("Enter the datetime in YYYYMMDD_HH_MM format:")
folder_name = raw_input()

start_date = datetime.datetime.strptime(folder_name, "%Y%m%d_%H_%M")
tec_resolution = datetime.timedelta(minutes=5)
#the folder name is the current datetime, so the predictions will be from the next hop based on the output freq 
start_date = start_date+tec_resolution*param.output_freq
print ("Baseline comparison for \"{}\" datetime".format(start_date))

#getting the datetime variables for the predicted TEC maps
datetime_list = [start_date]
for _ in range(1, param.num_of_output_tec_maps):
    time = datetime_list[-1]+tec_resolution*param.output_freq
    datetime_list.append(time)

#reading the numpy arrays
path = model_path+'/'+folder_name+'/'
pred = np.load(path+"pred.npy")
truth = np.load(path+"y.npy")

#transpose for iterating over the TEC maps
pred = np.transpose(pred, [2, 0, 1])
truth = np.transpose(truth, [2, 0, 1])

#TODO make sure the previous days TEC maps are present in the same folder
#taking previous days tec maps
dtm = start_date - datetime.timedelta(days=1)
baseline_dtm = [dtm]
baseline = []
for _ in range(0, param.num_of_output_tec_maps):
    filename = dtm.strftime("%Y%m%d") + "/" + dtm.strftime("%Y%m%d.%H%M") + ".npy"
    baseline.append(np.load(filename))
    dtm = dtm + tec_resolution*param.output_freq
    baseline_dtm.append(dtm)
    
baseline = np.array(baseline)
baseline_dtm = np.array(baseline_dtm)
formula = mpimg.imread("formula.jpg")

for i in range(0, param.num_of_output_tec_maps):
    #(1, 75, 73) -> (75, 73). Making it 2D for plotting
    pred1 = np.squeeze(pred[i])
    truth1 = np.squeeze(truth[i])
    baseline1 = np.squeeze(baseline[i])
    #for neglecting the missing values in the plot
    truth1 = np.ma.masked_where(truth1 <= -1, truth1)
    #setting it -0.5 (heuristic) for predicted maps
    pred1 = np.ma.masked_where(pred1 <= 0, pred1)
    
    baseline1 = np.ma.masked_where(baseline1 <= -1, baseline1)
    
    rel_diff_stresnet  = abs((pred1 - truth1)/truth1)
        
    rel_diff_baseline  = abs((baseline1 - truth1)/truth1) 
    
    
    fig = plt.figure(figsize=(25, 15))
    x = list(range(225, 360, 25)) + list(range(0, 35, 15)) 
    y = list(range(15, 90, 10))

    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)
    
    plt.suptitle(str(datetime_list[i]), fontsize=25)

    ax1.set_xticklabels(tuple(x), fontsize=18)
    ax1.set_yticklabels(y, fontsize=18)
    ax1.set_title('STResNet Prediction', fontsize=25)

    ax2.set_xticklabels(tuple(x), fontsize=18)
    ax2.set_yticklabels(y, fontsize=18)
    ax2.set_title('True TEC Map', fontsize=25)

    ax3.set_xticklabels(tuple(x), fontsize=18)
    ax3.set_yticklabels(y, fontsize=18)
    ax3.set_title('Baseline TEC Map ' + str(baseline_dtm[i]), fontsize=25)
    
    ax4.set_xticklabels(tuple(x), fontsize=18)
    ax4.set_yticklabels(y, fontsize=18)
    ax4.set_title('STResNet Error', fontsize=25)

    ax5.set_xticklabels(tuple(x), fontsize=18)
    ax5.set_yticklabels(y, fontsize=18)
    ax5.set_title('Baseline Error', fontsize=25)
    
    ax6.axis('off')
    
    
    im1 = ax1.pcolormesh(pred1, cmap='jet', vmin=0, vmax=20)
    im2 = ax2.pcolormesh(truth1, cmap='jet', vmin=0, vmax=20)
    im3 = ax3.pcolormesh(baseline1, cmap='jet', vmin=0, vmax=20)
    im4 = ax4.pcolormesh(rel_diff_stresnet, cmap='jet', vmin=0, vmax=1)
    im5 = ax5.pcolormesh(rel_diff_baseline, cmap='jet', vmin=0, vmax=1)
    im6 = ax6.imshow(formula)
    
    f = fig.colorbar(im1, ax=ax1)
    f.ax.set_yticklabels(f.ax.get_yticklabels(), fontsize=15)
    f = fig.colorbar(im2, ax=ax2)
    f.ax.set_yticklabels(f.ax.get_yticklabels(), fontsize=15)
    f = fig.colorbar(im3, ax=ax3)
    f.ax.set_yticklabels(f.ax.get_yticklabels(), fontsize=15)
    f = fig.colorbar(im4, ax=ax4)
    f.ax.set_yticklabels(f.ax.get_yticklabels(), fontsize=15)
    f = fig.colorbar(im5, ax=ax5)
    f.ax.set_yticklabels(f.ax.get_yticklabels(), fontsize=15)
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    fig.savefig(path+"bscmp"+str(i)+'.png', dpi=fig.dpi, bbox_inches='tight')
    
