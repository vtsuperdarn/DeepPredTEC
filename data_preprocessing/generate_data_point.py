import pandas as pd
import numpy as np
import sqlite3
import datetime as dt
import os

class tec_data_point():
    """A class to hold tec frames for a single data point for ST-ResNet"""

    def __init__(self, current_datetime,
                 file_dir="../data/tec_map/filled/",
                 tec_resolution=5,
                 closeness_freq=1, closeness_size=12,
                 period_freq=12, period_size=24,
                 trend_freq=36, trend_size=8):

	"""
	Parameters:
	----------
	
	closeness_freq : int
	    freq = 1 is tec_resolution mins
	closeness_size : int
	    size corresponds to the sample size. 
	period_freq : int
	period_size : int
	trend_freq : int
	trend_size : int

	""" 

	self.tec_resolution = tec_resolution
	self.current_datetime = current_datetime
	self.closeness_freq = closeness_freq
	self.closeness_size = closeness_size
	self.period_freq = period_freq
	self.period_size = period_size
	self.trend_freq = trend_freq
	self.trend_size = trend_size
	self.file_dir = file_dir
	self.data = self._get_data()

    def _get_data(self):

        # For future frame
        future_dtm = [self.current_datetime + dt.timedelta(minutes=self.tec_resolution)]     
        # For near frames
        near_dtm = [self.current_datetime - dt.timedelta(minutes=i*self.closeness_freq*self.tec_resolution)\
                    for i in range(self.closeness_size)]
        # For recent frames
        recent_dtm = [self.current_datetime - dt.timedelta(minutes=i*self.period_freq*self.tec_resolution)\
                    for i in range(self.period_size)]
        # For distant frames
        distant_dtm = [self.current_datetime - dt.timedelta(minutes=i*self.trend_freq*self.tec_resolution)\
                       for i in range(self.trend_size)]          
        # All frames
        dtms = future_dtm +  near_dtm + recent_dtm + distant_dtm

	data = []
        for dtm in dtms:
	    fname = self.file_dir + dtm.strftime("%Y%m%d") + "/" +\
		    dtm.strftime("%Y%m%d.%H%M") + ".npy"
            try:
                tec_map = np.load(fname)
                data.append(tec_map)
            except:
                data = None
                break
        if data is not None:
            data = np.array(data)

        return data
			

class tec_batch():

    """A class that holds a batch of data points for ST-ResNet"""

    def __init__(self, current_datetime, batch_size=32,
                 file_dir="../data/tec_map/filled/",
                 tec_resolution=5,
                 closeness_freq=1, closeness_size=12,
                 period_freq=12, period_size=24,
                 trend_freq=36, trend_size=8):

	self.batch_size=batch_size
	self.tec_resolution = tec_resolution
	self.current_datetime = current_datetime
	self.closeness_freq = closeness_freq
	self.closeness_size = closeness_size
	self.period_freq = period_freq
	self.period_size = period_size
	self.trend_freq = trend_freq
	self.trend_size = trend_size
	self.file_dir = file_dir
	self.data = self._get_data()

    def _get_data(self):

        dtms = [self.current_datetime + dt.timedelta(minutes=i*self.tec_resolution)\
                for i in range(self.batch_size)]

	data = []
        for dtm in dtms:
	    data_point = tec_data_point(dtm, file_dir=self.file_dir,
					tec_resolution=self.tec_resolution,
					closeness_freq=self.closeness_freq,
				 	closeness_size=self.closeness_size,
					period_freq=self.period_freq,
					period_size=self.period_size,
					trend_freq=self.trend_freq,
					trend_size=self.trend_size)

            if data_point.data is not None:
                data.append(data_point.data)
            else:
                data = None

        if data is not None:
            data = np.array(data)
	return data


if __name__ == "__main__":


    # initialize parameters
    current_datetime = dt.datetime(2015, 1, 2, 10, 5)
    file_dir="../data/tec_map/filled/"
    tec_resolution=5

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

    batch_size = 32    # Number of data_points

    data_point = tec_data_point(current_datetime,
                                file_dir=file_dir,
                                tec_resolution=tec_resolution,
                                closeness_freq=closeness_freq, closeness_size=closeness_size,
                                period_freq=period_freq, period_size=period_size,
                                trend_freq=trend_freq, trend_size=trend_size)
    if data_point.data is not None:
        print("The shape of a data point is " + str(data_point.data.shape))
    else:
        print("Not enough frames to construct a data point")


    batch = tec_batch(current_datetime, batch_size=batch_size,
                      file_dir=file_dir,
                      tec_resolution=tec_resolution,
                      closeness_freq=closeness_freq, closeness_size=closeness_size,
                      period_freq=period_freq, period_size=period_size,
                      trend_freq=trend_freq, trend_size=trend_size)

    if batch.data is not None:
        print("The shape of a batch is " + str(batch.data.shape))
    else:
        print("Not enough data_points to construct a batch")
