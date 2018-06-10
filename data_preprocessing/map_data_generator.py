'''
Run as python3 on amazon aws
Sampling algorithm for creating closeness, period and trend sets of TEC map
'''
import h5py
import numpy as np
import datetime

#storing the dictionaries into the list
file_dir = "/sd-data/backup/sd-guest/data/data_dict_ord/"
maps = []
for i in range(1, 75):
    data_dict = np.load(file_dir + "data_dict"+str(i)+".npy").item()
    maps.append(data_dict)
    del data_dict

npmaps = np.array(maps)    
print (npmaps.shape)


d1 = datetime.datetime(2015, 1, 1, 0, 5) 
d2 = datetime.datetime(2015, 1, 1, 0, 10)
start_date = datetime.datetime(2015, 1, 1, 0, 0) 
tec_resolution = (d2 - d1)
print (tec_resolution)
dic = npmaps[0]
print (len(dic.keys()))
#print (dic.keys()[-1])


all_tec_maps = []
all_datetime = []
#flag_start helps read the dictionaries in a continuous manner as if the days have been merged
flag_start = True
for data_dict in npmaps:    
    keys = data_dict.keys()
    #pad the start of the first day only
    if(flag_start):        
        prev_key = keys[0]
        curr_key = start_date
        while(curr_key != prev_key):
            all_tec_maps.append(data_dict[prev_key])
            all_datetime.append(curr_key) 
            curr_key += tec_resolution 
        all_tec_maps.append(data_dict[prev_key])
        all_datetime.append(prev_key)        
        start_index = 1
        flag_start = False
        #print len(all_tec_maps)
    else:
        start_index = 0
    
    for i in range(start_index, len(data_dict.keys())):
        next_key = keys[i]
        curr_key = prev_key + tec_resolution
        while(curr_key != next_key):
            all_tec_maps.append(all_tec_maps[-1])
            all_datetime.append(curr_key) 
            curr_key += tec_resolution
        all_tec_maps.append(data_dict[next_key])
        all_datetime.append(next_key) 
        prev_key = next_key


print (len(all_tec_maps))
all_tec_maps = np.array(all_tec_maps)
print (all_tec_maps.shape)



#closeness is sampled 12 times every 5 mins, lookback = (12*5min = 1 hour)
#freq 1 is 5mins
closeness_freq = 1
#size corresponds to the sample size
closeness_size = 12
#period is sampled 24 times every 1 hour (every 12th index), lookback = (24*12*5min = 1440min = 1day)
period_freq = 12
period_size = 24
#trend is sampled 24 times every 3 hours (every 36th index), lookback = (8*36*5min = 1440min = 1day)
trend_freq = 36
trend_size = 8
thresh = max(max(closeness_freq * closeness_size, period_freq * period_size), trend_freq * trend_size)
print ("thresh ", thresh)


Y = []
x_closeness = []
x_period = []
x_trend = []
file_no = 1
for i in range(thresh, all_tec_maps.shape[0]):
    #preparing indices to pick
    closeness_lookback = closeness_size * closeness_freq
    closeness_index = np.array(list(range(closeness_freq, closeness_lookback+1, closeness_freq))) * (-1) + i
    period_lookback = period_size * period_freq
    period_index = np.array(list(range(period_freq, period_lookback+1, period_freq))) * (-1) + i
    trend_lookback = trend_size * trend_freq
    trend_index = np.array(list(range(trend_freq, trend_lookback+1, trend_freq))) * (-1) + i
    #using fancy indexing from numpy    
    closeness = all_tec_maps[closeness_index].tolist()
    period = all_tec_maps[period_index].tolist()
    trend = all_tec_maps[trend_index].tolist()
    x_closeness.append(closeness)
    x_period.append(period)
    x_trend.append(trend)
    Y.append(all_tec_maps[i])
    if ((i-thresh) % 150 == 0 and i != thresh):
        with h5py.File("output_files/xcloseness_"+str(file_no)+".h5", 'w') as hf:
            hf.create_dataset("xcloseness_"+str(file_no),  data=x_closeness)    
        with h5py.File("output_files/xperiod_"+str(file_no)+".h5", 'w') as hf:
            hf.create_dataset("xperiod_"+str(file_no),  data=x_period)    
        with h5py.File("output_files/xtrend_"+str(file_no)+".h5", 'w') as hf:
            hf.create_dataset("xtrend_"+str(file_no),  data=x_trend)                                            
        with h5py.File("output_files/ydata_"+str(file_no)+".h5", 'w') as hf:
            hf.create_dataset("ydata_"+str(file_no),  data=Y)    
        x_closeness = []
        x_period = []
        x_trend = []
        Y = []         
        print ("Saving file " + str(file_no))
        file_no += 1
        
x_closeness = np.array(x_closeness)
Y = np.array(Y)


print (x_closeness.shape)
print (Y.shape)

with h5py.File("output_files/xcloseness_"+str(file_no)+".h5", 'w') as hf:
    hf.create_dataset("xcloseness_"+str(file_no),  data=x_closeness)    
with h5py.File("output_files/xperiod_"+str(file_no)+".h5", 'w') as hf:
    hf.create_dataset("xperiod_"+str(file_no),  data=x_period)    
with h5py.File("output_files/xtrend_"+str(file_no)+".h5", 'w') as hf:
    hf.create_dataset("xtrend_"+str(file_no),  data=x_trend)                                            
with h5py.File("output_files/ydata_"+str(file_no)+".h5", 'w') as hf:
    hf.create_dataset("ydata_"+str(file_no),  data=Y)
