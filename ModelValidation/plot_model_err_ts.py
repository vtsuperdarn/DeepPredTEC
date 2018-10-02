import datetime
import pandas
import numpy
import dask
import glob
import feather
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

class ModPerfTS(object):
    """
    A class to read in data from model files and generate
    a timeseries plot of average TEC value binned by latitude,
    to test the accuracy of the model.
    """
    def __init__(self, baseModelDir, modelName, modelDurtn, trueTecBaseDir,\
                 predStrtMinute=10, timeRange=None, latBinSize=10,\
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
        self.modelName = modelName
        self.modelDir = baseModelDir + modelName + "/" + "predicted_tec/"
        self.modelDurtn = modelDurtn # hours
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

    def get_err_data(self, remove_neg_tec_rows=True, modelType="deep"):
        """
        Generate mean and std in rel err as
        a time series in model pred
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
        trueTECDF = trueTECDF[ (trueTECDF["abs_tec_err"] > 0.) &\
                             (trueTECDF["true_tec"] > 0.) \
                             ].reset_index(drop=True)
        trueTECDF["rel_tec_err"] = \
                trueTECDF["abs_tec_err"]/trueTECDF["true_tec"]
        return trueTECDF

    def get_model_err_ts(self, downCastDF=True,\
             remove_neg_tec_rows=True, modelType="deep",\
              statType="median", errLatRange=None, \
              saveErrStatDF=True, errStatDir="../ErrorStats/"):
        """
        Generate mean and std in rel err as
        a time series in model pred
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
        trueTECDF = trueTECDF[ (trueTECDF["abs_tec_err"] > 0.) &\
                             (trueTECDF["true_tec"] > 0.) \
                             ].reset_index(drop=True)
        trueTECDF["rel_tec_err"] = \
                trueTECDF["abs_tec_err"]/trueTECDF["true_tec"]

        # Drop NaNs
        trueTECDF.dropna(inplace=True)
        # get the first prediction time!
        minTECDt64 = trueTECDF["date"].min()
        # this is in numpy datetime64, convert
        # to datetime object
        minTECTS64 = (minTECDt64 - \
              numpy.datetime64('1970-01-01T00:00:00Z')\
             ) / numpy.timedelta64(1, 's')
        minTECDate = datetime.datetime.utcfromtimestamp(\
                                        minTECTS64)
        # then we'll go to the next hour, else
        if minTECDate.minute == self.predStrtMinute:
            firstPredTime = minTECDate
        elif minTECDate.minute < self.predStrtMinute:
            firstPredTime = datetime.datetime(minTECDate.year,\
                                minTECDate.month, minTECDate.day,\
                                minTECDate.hour, self.predStrtMinute)
        else:
            firstPredTime = datetime.datetime(minTECDate.year,\
                                minTECDate.month, minTECDate.day,\
                                minTECDate.hour + 1, self.predStrtMinute)
        # only limit trueTECDF to timeperiods greater than firstPredTime
        trueTECDF = trueTECDF[ trueTECDF["date"] >= firstPredTime\
                             ].reset_index(drop=True)
        # get prediction minutes
        trueTECDF["pred_minute"] = trueTECDF.apply(\
                                       self.get_minutes_from_prediction,\
                                       args=(firstPredTime,), axis=1)
        # get time bins of predictions, i.e., 
        # if prediction duration is 4 hours divide hours
        # into bins 0-4, 4-8, 8-12 and so on...
        trueTECDF["binned_hours"] = [\
                    int(x.hour/self.modelDurtn)*self.modelDurtn for x\
                    in trueTECDF["date"] ]
        trueTECDF = trueTECDF.dropna().reset_index(drop=True)
        # Now groupby minutes from prediction and get mean and std errs
        if statType == "median":
            errStatDF = trueTECDF[ ["rel_tec_err", "pred_minute"]\
                                ].groupby(["pred_minute"]\
                                         ).median().reset_index()
            errStatDF = errStatDF.dropna().reset_index(drop=True)
            errStatDF.columns = [ "pred_minute", "mean_rel_err" ]
        else:
            errStatDF = trueTECDF[ ["rel_tec_err", "pred_minute"]\
                                ].groupby(["pred_minute"]\
                                         ).mean().reset_index()
            errStatDF = errStatDF.dropna().reset_index(drop=True)
            errStatDF.columns = [ "pred_minute", "mean_rel_err" ]
        errStdDF = trueTECDF[ ["rel_tec_err", "pred_minute"]\
                                ].groupby(["pred_minute"]\
                                         ).std().reset_index()
        errStdDF.columns = [ "pred_minute", "std_rel_err" ]
        # merge both median/mean and std DFs
        errStatDF = pandas.merge( errStatDF, errStdDF,\
                                 on=["pred_minute"] )
        # save the data (it takes a long time to calculate)
        if saveErrStatDF:
            errStatDF.to_csv(errStatDir + self.modelName + ".csv")
        return errStatDF
        
    def generate_ts_plots(self, figName, downCastDF=True,\
             remove_neg_tec_rows=True, errStatDir="./",\
             lgndFontSize='x-small', modelType="deep",\
              statType="median", errLatRange=None):
        """
        Generate mean and std relative error plots
        """
        errStatDF = self.get_model_err_ts(downCastDF=downCastDF,\
             remove_neg_tec_rows=remove_neg_tec_rows,\
             errStatDir=errStatDir, modelType=modelType,\
              statType=statType, errLatRange=errLatRange)
        # set seaborn styling
        sns.set_style("whitegrid")
        # set the fig!
        f = plt.figure(figsize=(12, 8))
        ax = f.add_subplot(1,1,1)
        ax.scatter(errStatDF['pred_minute'], errStatDF['mean_rel_err'],\
            marker='o', color='firebrick', alpha=0.7, s = 124)
        ax.errorbar(errStatDF['pred_minute'], errStatDF['mean_rel_err'],\
             yerr=errStatDF['std_rel_err'],  color='firebrick', label='',\
            capthick=2., capsize=5., fmt='o')
        ax.set_ylabel("Relative Error", fontsize=14)
        ax.set_xlabel("Minutes from Prediction", fontsize=14)
        ax.set_title( str(self.modelDurtn) + "- hour prediction" )
        plt.tick_params(labelsize=14)
        f.savefig(figName,bbox_inches='tight')

    def generate_err_dist_plot(self, figName, overlayQuartiles=True,\
                        pltErr="relative", lowerPercentile=10,\
                        upperPercentile=90, pltType="histogram"):
        """
        Generate error dist plots
        """
        from scipy.stats import gaussian_kde
        # get the error data
        trueTECDF = self.get_err_data()
        if pltErr == "relative":
            errArr = trueTECDF["rel_tec_err"].values
            xLabel = 'Relative TEC Error'
        else:
            errArr = trueTECDF["abs_tec_err"].values
            xLabel = 'Absolute TEC Error'
        # get the quartiles
        median, quar1, quar3 = numpy.percentile(errArr, 50),\
            numpy.percentile(errArr, lowerPercentile),\
             numpy.percentile(errArr, upperPercentile)
        # print "quar1, median, quar3 : ", quar1, median, quar3
        # set plot styling
        # sns.set_style("whitegrid")
        plt.style.use("fivethirtyeight")
        # set the fig!
        # f, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
        f = plt.figure(figsize=(12, 8))
        ax = f.add_subplot(1,1,1)
        if pltType == "histogram":
            # bins = [ 0, 0.2, 0.5, 1., 2., 4, 6., 8., 10. ]
            bins = numpy.linspace(0,1,100)#[ 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]#[ 0, 0.1, 0.2, 0.5, 1. ]
            # to overlay the quartiles get the freq at different bins
            errFreq, errBins= numpy.histogram(errArr, bins=bins)
            # Get the freq to plot
            plotFreqArr = [ numpy.percentile(errFreq, 68)/float(len(errArr)),\
                     numpy.percentile(errFreq, 72)/float(len(errArr)) ]
            plotFreqMed = numpy.percentile(errFreq, 70)/float(len(errArr))
            # plot the hist
            weights = numpy.ones_like(errArr)/float(len(errArr))
            ax.hist(errArr, bins=bins, color="#fc4f30",\
                     weights=weights, histtype="bar", lw=2.5, alpha=0.3)
            # Plot the percentile ranges
            # lower percentile
            ax.plot( [quar1, quar1], plotFreqArr, color="#008fd5" )
            # upper percentile
            ax.plot( [quar3, quar3], plotFreqArr, color="#008fd5" )
            # Mark the range
            ax.annotate(s='', xy=(quar3,plotFreqMed),\
                         xytext=(quar1,plotFreqMed),\
                        arrowprops=dict(arrowstyle='<->',color='#008fd5',
                                 lw=2.5,
                                 ls='--'))
        else:
            density = gaussian_kde(errArr)
            xs = numpy.linspace(0,1,100)
            # instantiate and fit the KDE model
            kde = gaussian_kde( errArr, bw_method="scott" )
            ax.plot( xs, kde(xs), color="#fc4f30" )
            # Plot the percentile ranges
            # lower percentile
            ax.plot( [quar1, quar1], [0.8, 1.2], color="#008fd5" )
            # upper percentile
            ax.plot( [quar3, quar3], [0.8, 1.2], color="#008fd5" )
            # Mark the range
            ax.annotate(s='', xy=(quar3,1.),\
                         xytext=(quar1,1.),\
                        arrowprops=dict(arrowstyle='<->',color='#008fd5',
                                 lw=2.5,
                                 ls='--'))
        # Plot the quartiles in a text box
        quar1Txt = str(lowerPercentile) + "th percentile"
        quar3Txt = str(upperPercentile) + "th percentile"
        textstr = '\n'.join((
                    quar1Txt + '$=%.2f$' % (quar1, ),
                    r'$\mathrm{median}=%.2f$' % (median, ),
                    quar3Txt + '$=%.2f$' % (quar3, )))
        # place a text box in upper right in axes coords
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', alpha=0.5)
        ax.text(0.65, 0.95, textstr, transform=ax.transAxes,\
                 fontsize=14, verticalalignment='top', bbox=props)
        ax.set_xticks(numpy.arange(0,1,0.1))
        ax.set_xlim( [0.,1.] )
        # Labeling
        plt.xlabel(xLabel)
        plt.ylabel('Density')
        plt.tick_params(labelsize=14)
        f.savefig(figName, bbox_inches='tight')
        
    def get_minutes_from_prediction(self, row, firstPredTime):
        """
        Given relative TEC error at different times
        estimate the minutes from first prediction.
        In other words, if a 2-hour prediction starts
        at 0 UT, then 0-2 UT would be 0-120 minutes, 
        2-4 UT would be another 0-120 minutes and so on.
        """
        return ((row["date"] - firstPredTime\
                    ).total_seconds()/60.)%(self.modelDurtn*60.)
    
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
