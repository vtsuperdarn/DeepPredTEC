import datetime
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import os 
import re
import glob


amean_err = []
astddev_err = []
amin_err = []
amax_err = []

rmean_err = []
rstddev_err = []
rmin_err = []
rmax_err = []

#loading the true and predicted tec maps for calculating the min/max error, mean and stddev error for both absolute and relative differences
for i in range(32):
    #print i
    path = "predicted_tec_files/{}_pred_*.npy".format(i)
    for fnm in glob.glob(path):
        pred = np.load(fnm).tolist()
    pred = np.array(pred)
    #print pred.shape


    path = "predicted_tec_files/{}_y_*.npy".format(i)
    for fnm in glob.glob(path):
        truth = np.load(fnm).tolist()
    truth = np.array(truth)
    #print truth.shape
    
    pred = np.squeeze(pred)
    truth = np.squeeze(truth)
    
    diff_absolute = abs(pred - truth)
    diff_relative = abs((pred - truth)/truth)
    #print diff.shape
    
    #flattern operation
    diff_absolute = np.reshape(diff_absolute, (32,-1))
    diff_relative = np.reshape(diff_relative, (32,-1))
    
    #print diff.shape
    amean_err += np.mean(diff_absolute, axis=1).tolist()
    astddev_err += np.std(diff_absolute, axis=1).tolist()
    amin_err += np.min(diff_absolute, axis=1).tolist()
    amax_err += np.max(diff_absolute,axis=1).tolist()
    
    rmean_err += np.mean(diff_relative, axis=1).tolist()
    rstddev_err += np.std(diff_relative, axis=1).tolist()
    rmin_err += np.min(diff_relative, axis=1).tolist()
    rmax_err += np.max(diff_relative,axis=1).tolist()
    
#starting from 168 because we want one day cycle plot    
amean_err = amean_err[168:]
astddev_err = astddev_err[168:]
amin_err = amin_err[168:]
amax_err = amax_err[168:]

rmean_err = rmean_err[168:]
rstddev_err = rstddev_err[168:]
rmin_err = rmin_err[168:]
rmax_err = rmax_err[168:]


amean_err = np.array(amean_err)
astddev_err = np.array(astddev_err)
amin_err = np.array(amin_err)
amax_err = np.array(amax_err)
print amean_err.shape
print astddev_err.shape
print amin_err.shape
print amax_err.shape


rmean_err = np.array(rmean_err)
rstddev_err = np.array(rstddev_err)
rmin_err = np.array(rmin_err)
rmax_err = np.array(rmax_err)
print rmean_err.shape
print rstddev_err.shape
print rmin_err.shape
print rmax_err.shape


#plotting the absolute error plots
sns.set_style("whitegrid")
sns.set_context("poster")
f, axArr = plt.subplots(5, sharex=True, figsize=(20, 20))
xlim1 = amean_err.shape[0]
dates = []
stdate = datetime.datetime(2015, 1, 12, 0, 5) 
dummy = datetime.datetime(2015, 1, 12, 0, 10)
tec_resolution = (dummy - stdate)
dates.append(stdate)
for i in range(1, 856):
    dates.append(dates[i-1]+tec_resolution)

x_val = dates
print len(x_val)
cl = sns.color_palette('bright', 4)
axArr[0].plot(x_val, amean_err, color=cl[0])
axArr[1].plot(x_val, astddev_err, color=cl[1])
axArr[2].plot(x_val, amin_err, color=cl[2])
axArr[3].plot(x_val, amax_err, color=cl[3])
axArr[4].plot(x_val, amean_err, color=cl[0], label='mean')
axArr[4].plot(x_val, astddev_err, color=cl[1], label='stddev')


axArr[0].set_ylabel("Mean", fontsize=14)
axArr[1].set_ylabel("Stddev", fontsize=14)
axArr[2].set_ylabel("Min", fontsize=14)
axArr[3].set_ylabel("Max", fontsize=14)
axArr[4].set_ylabel("Mean/Stddev", fontsize=14)
axArr[-1].set_xlabel("TIME", fontsize=14)


axArr[0].get_xaxis().set_major_formatter(DateFormatter('%H:%M'))
axArr[1].get_xaxis().set_major_formatter(DateFormatter('%H:%M'))
axArr[2].get_xaxis().set_major_formatter(DateFormatter('%H:%M'))
axArr[3].get_xaxis().set_major_formatter(DateFormatter('%H:%M'))
axArr[4].get_xaxis().set_major_formatter(DateFormatter('%H:%M'))

axArr[4].legend( bbox_to_anchor=(0., 1.02, 1., .102), loc=1, ncol=2, borderaxespad=0.1 )
f.savefig('error_plot_absolute.png', dpi=f.dpi, bbox_inches='tight')


#plotting the relative error plots
sns.set_style("whitegrid")
sns.set_context("poster")
f, axArr = plt.subplots(5, sharex=True, figsize=(20, 20))
xlim1 = rmean_err.shape[0]
dates = []
stdate = datetime.datetime(2015, 1, 12, 0, 5) 
dummy = datetime.datetime(2015, 1, 12, 0, 10)
tec_resolution = (dummy - stdate)
dates.append(stdate)
for i in range(1, 856):
    dates.append(dates[i-1]+tec_resolution)

x_val = dates
print len(x_val)
cl = sns.color_palette('bright', 4)
axArr[0].plot(x_val, rmean_err, color=cl[0])
axArr[1].plot(x_val, rstddev_err, color=cl[1])
axArr[2].plot(x_val, rmin_err, color=cl[2])
axArr[3].plot(x_val, rmax_err, color=cl[3])
axArr[4].plot(x_val, rmean_err, color=cl[0], label='mean')
axArr[4].plot(x_val, rstddev_err, color=cl[1], label='stddev')

axArr[0].set_ylabel("Mean", fontsize=14)
axArr[1].set_ylabel("Stddev", fontsize=14)
axArr[2].set_ylabel("Min", fontsize=14)
axArr[3].set_ylabel("Max", fontsize=14)
axArr[4].set_ylabel("Mean/Stddev", fontsize=14)
axArr[-1].set_xlabel("TIME", fontsize=14)


axArr[0].get_xaxis().set_major_formatter(DateFormatter('%H:%M'))
axArr[1].get_xaxis().set_major_formatter(DateFormatter('%H:%M'))
axArr[2].get_xaxis().set_major_formatter(DateFormatter('%H:%M'))
axArr[3].get_xaxis().set_major_formatter(DateFormatter('%H:%M'))
axArr[4].get_xaxis().set_major_formatter(DateFormatter('%H:%M'))

axArr[4].legend( bbox_to_anchor=(0., 1.02, 1., .102), loc=1, ncol=2, borderaxespad=0.1 )
f.savefig('error_plot_relative.png', dpi=f.dpi, bbox_inches='tight')
