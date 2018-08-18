import datetime
import numpy
import dask
import collections


class BatchDateUtils(object):
    
    """
    Instead of caclulating batch dates
    and corresponding data point dates
    on the fly, we'll pre-calculate them
    and store them in a dict
    """
    
    def __init__(self, start_date, end_date, batch_size, resolution,\
                closeness_freq, closeness_size, period_freq, period_size,\
                trend_freq, trend_size, num_outputs, output_freq):
        """
        set up the parameters
        """
        self.start_date = start_date
        self.end_date = end_date
        self.batch_size = batch_size
        self.resolution = resolution
        self.closeness_freq = closeness_freq
        self.period_freq = period_freq
        self.trend_freq = trend_freq
        self.closeness_size = closeness_size
        self.period_size = period_size
        self.trend_size = trend_size
        self.num_outputs = num_outputs #this defines the num of output tec maps 
        self.output_freq = output_freq
        self.batch_dict = self._get_batch_dict()

    def _get_batch_dict(self):
        """
        create a dict with batch dates as keys
        and corresponding datapoint date list as
        values.
        """
        batch_dict = collections.OrderedDict()
        batch_curr_date = self.start_date
        while batch_curr_date <= self.end_date:
            batch_dict[batch_curr_date] = self.get_data_point_arrays(batch_curr_date)
            batch_curr_date += datetime.timedelta(minutes=self.resolution*self.batch_size)
        return batch_dict

    def get_data_point_arrays(self, curr_time):
        """
        Generate the times (for near, recent, distant and future)
        channels
        """
        # loop through all the data points in the batch and construct arrays
        dtms = [curr_time + datetime.timedelta(minutes=i*self.resolution)\
                for i in range(self.batch_size)]
        dp_dict = collections.OrderedDict()
        for dtm in dtms :
            sub_dp_dict = collections.OrderedDict()
            # For future frame
            sub_dp_dict['future_dtm'] = [dtm + datetime.timedelta(minutes=i*self.output_freq*self.resolution)\
                        for i in range(1, self.num_outputs+1)]     
            # For near frames
            sub_dp_dict['near_dtm'] = [dtm - datetime.timedelta(minutes=i*self.closeness_freq*self.resolution)\
                        for i in range(self.closeness_size)]
            # For recent frames
            sub_dp_dict['recent_dtm'] = [dtm - datetime.timedelta(minutes=i*self.period_freq*self.resolution)\
                        for i in range(self.period_size)]
            # For distant frames
            sub_dp_dict['distant_dtm'] = [dtm - datetime.timedelta(minutes=i*self.trend_freq*self.resolution)\
                           for i in range(self.trend_size)]
            dp_dict[dtm] = sub_dp_dict
        return dp_dict


class TECUtils(object):
    """
    Loading TEC data and creating batches
    in bulk
    """
    def __init__(self, start_date, end_date, tec_dir, tec_resolution, load_window):
        """
        set up parameters and data
        """
        # subtract 7 days from start date (and vice versa for end data)
        # because these dates estimate time backwards and TEC
        # needs to load for these days.
        self.start_date = start_date -  datetime.timedelta(days=load_window)
        self.end_date = end_date + datetime.timedelta(days=load_window)
        self.tec_dir = tec_dir
        self.tec_resolution = tec_resolution
        # If you don't want to use dask comment the lines below
        self.tec_data = {}
        self._dask_bulk_load_tec()
        # and uncomment the lines below
        #self.tec_data = self._bulk_load_tec()

    def _bulk_load_tec(self):
        """
        The name makes it pretty obvious
        """
        tec_data_dict = {}
        # Loop through each date and load the corresponding TEC file
        curr_date = self.start_date
        while curr_date <= self.end_date:
            fname = self.tec_dir + curr_date.strftime("%Y%m%d") + "/" +\
                    curr_date.strftime("%Y%m%d.%H%M") + ".npy"
            tec_data_dict[curr_date] = numpy.load(fname).transpose()
            curr_date += datetime.timedelta(minutes=self.tec_resolution)

        return tec_data_dict

    def _dask_bulk_load_tec(self):
        """
        The name makes it pretty obvious
        """
        # Loop through each date and load the corresponding TEC file
        curr_date = self.start_date
        while curr_date <= self.end_date:
            #creating dask variables and storing in the tec_data dict
            self.load_npy_file(curr_date)
            curr_date += datetime.timedelta(minutes=self.tec_resolution)
        
        #computing the actual values of the dictionary - values are the numpy.load calls
        self.tec_data = dask.delayed(self.tec_data).compute()
        
   
    def load_npy_file(self,curr_date):
        """
        Load a correponding TEC file given a date
        and transpose the result
        """
        fname = self.tec_dir + curr_date.strftime("%Y%m%d") + "/" +\
                    curr_date.strftime("%Y%m%d.%H%M") + ".npy"
        self.tec_data[curr_date] = dask.delayed(numpy.load)(fname).transpose()


    def create_batch(self, batch_time_dict):
        """
        Given a batch_time_dict return a tec batch data point
        """
        data_close = []
        data_period = []
        data_trend = []
        data_out = []
        # Loop through all the data points and get the data
        for dp_time in batch_time_dict.keys():
            dp_time_dict = batch_time_dict[dp_time]
            data_close.append( numpy.array( [ self.tec_data[k] for k in\
                         dp_time_dict['near_dtm'] ] ).transpose() )
            data_period.append( numpy.array( [ self.tec_data[k] for k in\
                         dp_time_dict['recent_dtm'] ] ).transpose() )
            data_trend.append( numpy.array( [ self.tec_data[k] for k in\
                         dp_time_dict['distant_dtm'] ] ).transpose() )
            data_out.append( numpy.array( [ self.tec_data[k] for k in\
                         dp_time_dict['future_dtm'] ] ).transpose() )
        return ( numpy.array(data_close), numpy.array(data_period),\
                     numpy.array(data_trend), numpy.array(data_out) )

