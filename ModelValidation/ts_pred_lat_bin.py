import datetime
import pandas
import numpy
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt

class ModValTSLat(object):
    """
    A class to read in data from model files and generate
    a timeseries plot of average TEC value binned by latitude,
    to test the accuracy of the model.
    """
    def __init__(self, baseDir, modelName, timeRange=None,\
             latBinSize=10, mlonRange=None):
        """
        baseDir : parent dir where all models are stored
        modelName : name of the model being tested
        timeRange : time range of the plot. If set to None
                    the time range is determined by the 
                    files in the directory. Otherwise, we
                    expect a two element datetime object list!
        latBinSize : size of the latitude bin over which
                     tec values are averaged.
        mlonRange : range of mlons over which TEC values are averaged
                    If set to None all MLONs are used.
        """
        self.modelDir = baseDir + modelName + "/" + "predicted_tec/"
        self.timeRange = timeRange
        # Make sure the start time and end time minute are multiple of 0 or 5
        if self.timeRange is not None:
            assert self.timeRange[0].minute%5 == 0,\
                "Start Time minute should end with 0 or 5."
            assert self.timeRange[1].minute%5 == 0,\
                "End Time minute should end with 0 or 5."
        self.latBinSize = latBinSize
        self.mlonRange = mlonRange

    def read_data(self, fType="pred"):
        """
        Read data from the npy files
        fType : Type of file to read (either pred or true!)
        """
        # depending on the time range and fType get a list
        # of fileNames that need to be loaded!
        # Get a list of all the files in the dir!
        fList = glob.glob( self.modelDir + "*" + fType + ".npy" )
        if self.timeRange is not None:
            # If a timerange is set generate names of all possible files
            # We know that TEC data has a time resolution of 5 min! we'll use 
            # that information to generate all possible file names! then use 
            # set intersection operation to choose only those files which fall
            # with in the time range. This I feel is a easy way to work with 
            # variable time resolution for the predictions.
            fNameList = []
            currTime = self.timeRange[0]
            while currTime <= self.timeRange[1]:
                dtStr = currTime.strftime("%Y%m%d.%H%M")
                fNameList.append( self.modelDir + dtStr + "_" +\
                             fType + ".npy" )
                currTime += datetime.timedelta(seconds=5*60)
            # intersection of both the lists
            fList = list( set(fList).intersection(set(fNameList)) )
            
        # _tecData = numpy.load(_fName).transpose()


if __name__ == "__main__":

    modelName = "model_batch64_epoch100_resnet100_nresfltr24_nfltr12_of2_otec24_cf2_csl72_pf12_psl72_tf36_tsl8_gs32_ks55_exoT_nrmT_yr_11_13_310.1902163028717_values"
    baseDir = "/sd-data/DeepPredTEC/ModelValidation/"
    timeRange = [ datetime.datetime(2015,3,5), datetime.datetime(2015,3,10) ]
    tsObj = ModValTSLat(baseDir, modelName, timeRange=timeRange)
    tsObj.read_data()