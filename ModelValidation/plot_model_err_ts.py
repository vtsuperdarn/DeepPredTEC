import datetime
import pandas
import numpy
import dask
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

class ModPerfTS(object):
    """
    A class to read in data from model files and generate
    a timeseries plot of average TEC value binned by latitude,
    to test the accuracy of the model.
    """
    def __init__(self, baseModelDir, modelName, modelDurtn, predStrtMinute=0,\
                 trueTecBaseDir, timeRange=None, latBinSize=10,\
                 mlonRange=None, useMask=True,\
                 maskFile="../WeightMatrix/w2_mask-2011-2013-80perc.npy"):
        """
        baseModelDir : parent dir where all models are stored
        modelName : name of the model being tested
        modelDurtn : num. of hours predicted
        predStrtMinute : minute at which predictiong started.
        timeRange : time range of the plot. If set to None
                    the time range is determined by the 
                    files in the directory. Otherwise, we
                    expect a two element datetime object list!
        latBinSize : size of the latitude bin over which
                     tec values are averaged.
        mlonRange : range of mlons over which TEC values are averaged
                    If set to None all MLONs are used.
        """
        self.modelDir = baseModelDir + modelName + "/" + "predicted_tec/"
        self.modelDurtn = modelDurtn
        self.predStrtMinute = predStrtMinute
        self.trueTecBaseDir = trueTecBaseDir
        self.timeRange = timeRange
        # Make sure the start time and end time minute are multiple of 0 or 5
        if self.timeRange is not None:
            assert self.timeRange[0].minute%5 == 0,\
                "Start Time minute should end with 0 or 5."
            assert self.timeRange[1].minute%5 == 0,\
                "End Time minute should end with 0 or 5."
        self.latBinSize = latBinSize
        self.mlonRange = mlonRange
        self.useMask= useMask
        # if masking is set to true read the mask file
        if self.useMask :
            self.maskMat = numpy.load(maskFile)
        self.tecModelDict = {}
        self.tecTrueDict = {}

    def read_dl_model_true_data(self, refInpDir="/sd-data/med_filt_tec/",\
                    refFileDate=datetime.datetime(2015,1,1), fType="pred"):
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
        # Load the actual TEC maps
        for _fn in fList:
            cfn = _fn.split("/")[-1]
            fDate = cfn.split(".")[0]
            fTime = cfn.split(".")[1].split("_")[0]
            cd = datetime.datetime.strptime(fDate +\
                                "-" + fTime, "%Y%m%d-%H%M")
            # get the fName for actual data dir
            tfn = self.trueTecBaseDir + fDate + "/" + fDate +\
                         "." + fTime + ".npy"
            # Load the actual data
            self.load_npy_file(cd, _fn, "pred")
            self.load_npy_file(cd, tfn, "true")
        # read a dummy tec file into pandas DF to convert the numpy
        # files into a DF with appropriate columns
        # Read the median filtered TEC data
        inpColList = [ "dateStr", "timeStr", "Mlat",\
                       "Mlon", "med_tec", "dlat", "dlon" ]
        inpRefFile = refInpDir + "tec-medFilt-" +\
                     refFileDate.strftime("%Y%m%d") + ".txt"
        dfRef = pandas.read_csv(inpRefFile, delim_whitespace=True,
                         header=None, names=inpColList)
        # Change Mlon range from 0-360 to -180 to 180
        dfRef.loc[:, "Mlon"] = dfRef.Mlon.apply(lambda x: x if\
                                     x<=180 else x-360)
        mlat_min, mlon_west, mlon_east=15., -110, 34.
        testTimeStr = 1200
        dfRef = dfRef[ (dfRef["timeStr"] == testTimeStr) &
            (dfRef["Mlat"] >= mlat_min) &\
            (dfRef["Mlon"] >= mlon_west) &\
            (dfRef["Mlon"] <= mlon_east) ].reset_index(drop=True)
        # pivot dfRef to get the cols
        dfRef = dfRef.pivot(index="Mlat", columns="Mlon",\
                 values="med_tec")
        # Predicted DF
        predDFList = []
        for _tdk in self.tecModelDict.keys():
            dfRef[dfRef.columns] = self.tecModelDict[_tdk]
            # unpivot the DF
            ndf = dfRef.unstack().reset_index(name='med_tec')
            # add a date col
            ndf["date"] = _tdk
            predDFList.append( ndf )
        # append the DFs
        predTECDF = pandas.concat(predDFList).sort_values(\
                        "date").reset_index(drop=True)
        trueDFList = []
        for _tdk in self.tecTrueDict.keys():
            dfRef[dfRef.columns] = self.tecTrueDict[_tdk]
            # unpivot the DF
            ndf = dfRef.unstack().reset_index(name='med_tec')
            # add a date col
            ndf["date"] = _tdk
            trueDFList.append( ndf )
        # append the DFs
        trueTECDF = pandas.concat(trueDFList).sort_values(\
                        "date").reset_index(drop=True)
        return (predTECDF, trueTECDF)

    def read_baseline_model_true_data(self, refInpDir="/sd-data/med_filt_tec/",\
                    refFileDate=datetime.datetime(2015,1,1)):
        """
        Read data from the npy files
        """
        # Get a list of all the files in the deeplearning predicted tec dir!
        # NOTE we are reading file dates from deeplearning model
        # since we need a 1:1 comparison. So even for the baseline
        # model we'll only use those datetimes where we have model
        # outputs (i.e., same time resolution).
        fList = glob.glob( self.modelDir + "*" + "pred" + ".npy" )
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
                             "pred" + ".npy" )
                currTime += datetime.timedelta(seconds=5*60)
            # intersection of both the lists
            fList = list( set(fList).intersection(set(fNameList)) )
        # Load the actual TEC maps
        for _fn in fList:
            cfn = _fn.split("/")[-1]
            fDate = cfn.split(".")[0]
            fTime = cfn.split(".")[1].split("_")[0]
            cd = datetime.datetime.strptime(fDate +\
                                "-" + fTime, "%Y%m%d-%H%M")
            pd = cd - datetime.timedelta(days=1)
            pfDate = pd.strftime("%Y%m%d")
            pfTime = pd.strftime("%H%M")
            # get the fName for actual data dir
            tfn = self.trueTecBaseDir + fDate + "/" + fDate +\
                         "." + fTime + ".npy"
            pfn = self.trueTecBaseDir + pfDate + "/" + pfDate +\
                         "." + pfTime + ".npy"
            # Load the actual data
            self.load_npy_file(cd, pfn, "pred")
            self.load_npy_file(cd, tfn, "true")
        # read a dummy tec file into pandas DF to convert the numpy
        # files into a DF with appropriate columns
        # Read the median filtered TEC data
        inpColList = [ "dateStr", "timeStr", "Mlat",\
                       "Mlon", "med_tec", "dlat", "dlon" ]
        inpRefFile = refInpDir + "tec-medFilt-" +\
                     refFileDate.strftime("%Y%m%d") + ".txt"
        dfRef = pandas.read_csv(inpRefFile, delim_whitespace=True,
                         header=None, names=inpColList)
        # Change Mlon range from 0-360 to -180 to 180
        dfRef.loc[:, "Mlon"] = dfRef.Mlon.apply(lambda x: x if\
                                     x<=180 else x-360)
        mlat_min, mlon_west, mlon_east=15., -110, 34.
        testTimeStr = 1200
        dfRef = dfRef[ (dfRef["timeStr"] == testTimeStr) &
            (dfRef["Mlat"] >= mlat_min) &\
            (dfRef["Mlon"] >= mlon_west) &\
            (dfRef["Mlon"] <= mlon_east) ].reset_index(drop=True)
        # pivot dfRef to get the cols
        dfRef = dfRef.pivot(index="Mlat", columns="Mlon",\
                 values="med_tec")
        # Predicted DF
        predDFList = []
        for _tdk in self.tecModelDict.keys():
            dfRef[dfRef.columns] = self.tecModelDict[_tdk]
            # unpivot the DF
            ndf = dfRef.unstack().reset_index(name='med_tec')
            # add a date col
            ndf["date"] = _tdk
            predDFList.append( ndf )
        # append the DFs
        predTECDF = pandas.concat(predDFList).sort_values(\
                        "date").reset_index(drop=True)
        trueDFList = []
        for _tdk in self.tecTrueDict.keys():
            dfRef[dfRef.columns] = self.tecTrueDict[_tdk]
            # unpivot the DF
            ndf = dfRef.unstack().reset_index(name='med_tec')
            # add a date col
            ndf["date"] = _tdk
            trueDFList.append( ndf )
        # append the DFs
        trueTECDF = pandas.concat(trueDFList).sort_values(\
                        "date").reset_index(drop=True)
        return (predTECDF, trueTECDF)

    def generate_ts_plots(self, figName, downCastDF=True,\
             remove_neg_tec_rows=True, legenNcols=4,\
             lgndFontSize='x-small', modelType="deep",\
              statType="median", errLatRange=None):
        """
        Generate plots based on the input conditions
        (1) remove_neg_tec_rows : remove all the rows where tec 
                                  values are negative!
        """
        if modelType == "deep":
            (predTECDF, trueTECDF) = self.read_dl_model_true_data()
        else:
            (predTECDF, trueTECDF) = self.read_baseline_model_true_data()
        # Downcast Mlon, Mlat and med_tec to float16's, this way
        # we reduce the space occupied by the DF by almost 1/2
        predTECDF["Mlon"] = predTECDF["Mlon"].astype(numpy.float16)
        predTECDF["Mlat"] = predTECDF["Mlat"].astype(numpy.float16)
        predTECDF["med_tec"] = predTECDF["med_tec"].astype(numpy.float16)
        # Same for True TEC DF
        trueTECDF["Mlon"] = trueTECDF["Mlon"].astype(numpy.float16)
        trueTECDF["Mlat"] = trueTECDF["Mlat"].astype(numpy.float16)
        trueTECDF["med_tec"] = trueTECDF["med_tec"].astype(numpy.float16)
        # remove all rows where TEC values are negative
        if remove_neg_tec_rows:
            predTECDF.loc[(predTECDF['med_tec'] < 0), 'med_tec']=numpy.nan
            predTECDF = predTECDF.dropna()
            # same true tec df
            trueTECDF.loc[(trueTECDF['med_tec'] < 0), 'med_tec']=numpy.nan
            trueTECDF = trueTECDF.dropna()
        if self.mlonRange is not None:
            pretTECDF = pretTECDF[ (\
                     pretTECDF["Mlon"] >= self.mlonRange[0] ) &\
                     (pretTECDF["Mlon"] <= self.mlonRange[1] ) ]
            # same for true df
            trueTECDF = trueTECDF[ (\
                     trueTECDF["Mlon"] >= self.mlonRange[0] ) &\
                     (trueTECDF["Mlon"] <= self.mlonRange[1] ) ]
        # Divide the Mlats in the DF into bins
        minLat = int(self.latBinSize * round(\
                    predTECDF["Mlat"].min()/self.latBinSize))
        maxLat = int(self.latBinSize * round(\
                    predTECDF["Mlat"].max()/self.latBinSize))
        latBins = range(minLat, maxLat + self.latBinSize, self.latBinSize)
        colList = predTECDF.columns
        predTECDF = pandas.concat( [ predTECDF, \
                            pandas.cut( predTECDF["Mlat"], \
                                       bins=latBins ) ], axis=1 )
        predTECDF.columns = list(colList) + ["mlat_bins"]
        predTECDF.rename(columns={'med_tec': 'pred_tec'},\
                         inplace=True)
        # same for true tec df
        trueTECDF = pandas.concat( [ trueTECDF, \
                            pandas.cut( trueTECDF["Mlat"], \
                                       bins=latBins ) ], axis=1 )
        trueTECDF.columns = list(colList) + ["mlat_bins"]
        trueTECDF.rename(columns={'med_tec': 'true_tec'},\
                         inplace=True)
        trueTECDF = pandas.merge( trueTECDF, predTECDF,\
                                 on=["date", "Mlat",\
                                     "Mlon", "mlat_bins"] )
        trueTECDF["abs_tec_err"] = numpy.abs(\
                 trueTECDF["pred_tec"] - trueTECDF["true_tec"] )
        trueTECDF["rel_tec_err"] = \
                trueTECDF["abs_tec_err"]/trueTECDF["true_tec"]
        # Drop NaNs
        trueTECDF.dropna(inplace=True)
        trueTECDF["hour"] = [ x.hour for x\
                                 in trueTECDF["date"] ]
        trueTECDF["minute"] = [ x.dayofyear for x\
                               in trueTECDF["date"] ]
        # We need to make a plot of minutes from
        print trueTECDF.head()
    
    def load_npy_file(self, currDate, fName, fType):
        """
        Load a correponding TEC file into the dict
        """
        if fType == "pred":
            if self.useMask :
                self.tecModelDict[currDate] = numpy.load(fName) * self.maskMat
            else:
                self.tecModelDict[currDate] = numpy.load(fName)
        else:
            if self.useMask :
                self.tecTrueDict[currDate] = numpy.load(fName) * self.maskMat
            else:
                self.tecTrueDict[currDate] = numpy.load(fName)

if __name__ == "__main__":

    modelName = "model_batch64_epoch100_resnet100_nresfltr12_nfltr12_of2_otec12_cf2_csl72_pf12_psl72_tf36_tsl8_gs32_ks55_exoT_nrmT_w0_yr_11_13_379.3419065475464_values"
    baseModelDir = "/sd-data/DeepPredTEC/ModelValidation/"
    trueTecBaseDir = "/sd-data/DeepPredTEC/data/tec_map/filled/"
    modelDurtn = 2
    timeRange = [ datetime.datetime(2015,3,5), datetime.datetime(2015,3,10) ]
    tsObj = ModPerfTS(baseModelDir, modelName,\
             trueTecBaseDir, timeRange=timeRange)
    figName = "/home/bharat/Desktop/marc-examples/t/mod-err.pdf"
    tsObj.generate_ts_plots(figName)


