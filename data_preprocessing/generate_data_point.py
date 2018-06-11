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

        dtms = [self.current_datetime + dt.timedelta(minutes=self.tec_resolution)] +\
               [self.current_datetime - dt.timedelta(minutes=i*self.closeness_freq*self.tec_resolution)\
                for i in range(self.closeness_size)] +\
               [self.current_datetime - dt.timedelta(minutes=i*self.period_freq*self.tec_resolution)\
                for i in range(self.period_size)] +\
               [self.current_datetime - dt.timedelta(minutes=i*self.trend_freq*self.tec_resolution)\
                for i in range(self.trend_size)]

	data = []
        for dtm in dtms:
	    fname = self.file_dir + dtm.strftime("%Y%m%d") + "/" +\
		    dtm.strftime("%Y%m%d.%H%M") + ".npy"
	    data.append(np.load(fname))

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

	    data.append(data_point)

	return data


if __name__ == "__main__":


    # initialize parameters
    current_datetime = dt.datetime(2015, 1, 2, 10, 5)

    data_point = tec_data_point(current_datetime,
                                file_dir="../data/tec_map/filled/",
                                tec_resolution=5,
                                closeness_freq=1, closeness_size=12,
                                period_freq=12, period_size=24,
                                trend_freq=36, trend_size=8)

    batch = tec_batch(batch_size=32, current_datetime,
                      file_dir="../data/tec_map/filled/",
                      tec_resolution=5,
                      closeness_freq=1, closeness_size=12,
                      period_freq=12, period_size=24,
                      trend_freq=36, trend_size=8)

