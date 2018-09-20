import datetime
import pandas
import numpy
import dask
import glob
import seaborn as sns
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.colors import ListedColormap
from davitpy import utils

class DatePlots(object):
    """
    A class to read in predicted TEC map files for
    given timeRange and plot them.
    """
    def __init__(self, baseModelDir, modelName,\
             plotTimeRange, timeInterval, useMask=True,\
             maskFile="../WeightMatrix/w2_mask-2011-2013-80perc.npy"):
        """
        baseModelDir : parent dir where all models are stored
        modelName : name of the model being tested
        plotTimeRange : time range for the plots.
        timeInterval : time interval between plots in hours.
        useMask : Use mask to remove unwanted datapoints
        maskFile : maskfile to use, if we choose to use a mask.
        """
        self.modelDir = baseModelDir + modelName + "/" + "predicted_tec/"
        self.plotTimeRange = plotTimeRange
        self.timeInterval = timeInterval
        # Make sure the start time and end time minute are multiple of 10
        assert self.plotTimeRange[0].minute%10 == 0,\
            "Start Time minute should end with 0 or 5."
        assert self.plotTimeRange[1].minute%10 == 0,\
            "End Time minute should end with 0 or 5."
        self.useMask= useMask
        # if masking is set to true read the mask file
        if self.useMask :
            self.maskMat = numpy.load(maskFile)
        self.tecModelDict = OrderedDict()
        self.tecTrueDict = OrderedDict()
        self.tecBaselineDict = OrderedDict()

    def read_mask_file(self):
        """
        Given a mask filename read the data
        """
        # Read data from the mask file
        return numpy.load(self.maskFile).transpose()

    def load_pred_tec_data(self):
        """
        Read data from the predicted tec files
        """
        # get a list of times to read data from
        _ct = self.plotTimeRange[0]
        while _ct <= self.plotTimeRange[1]:
            # get the name of the file to be read
            _modfn = self.modelDir + _ct.strftime("%Y%m%d") + "." +\
                     _ct.strftime("%H%M") + "_pred.npy"
            # Load the actual data
            self.load_npy_file(_ct, _modfn, "pred")
            _ct += datetime.timedelta(hours=self.timeInterval)

    def load_true_tec_data(self,\
         trueTecBaseDir = "/sd-data/DeepPredTEC/data/tec_map/filled/"):
        """
        Read data from the actual tec files
        """
        # get a list of times to read data from
        _ct = self.plotTimeRange[0]
        while _ct <= self.plotTimeRange[1]:
            # get the name of the file to be read
            _modfn = self.modelDir + _ct.strftime("%Y%m%d") + "." +\
                     _ct.strftime("%H%M") + "_pred.npy"
            # get the fName for actual data dir
            # Load the actual data
            _trufn = trueTecBaseDir + _ct.strftime("%Y%m%d") +\
                         "/" + _ct.strftime("%Y%m%d") +\
                         "." + _ct.strftime("%H%M") + ".npy"
            self.load_npy_file(_ct, _trufn, "true")
            _ct += datetime.timedelta(hours=self.timeInterval)
            
    def load_baseline_tec_data(self,\
         trueTecBaseDir = "/sd-data/DeepPredTEC/data/tec_map/filled/"):
        """
        Read baseline tec data. This is nothing but the
        actual tec data from the previous day!
        """
        # get a list of times to read data from
        _ct = self.plotTimeRange[0]
        while _ct <= self.plotTimeRange[1]:
            # get the results from the previous day
            _pt = _ct - datetime.timedelta(days=1)
            # get the fName for actual data dir
            # Load the actual data
            _bslnfn = trueTecBaseDir + _pt.strftime("%Y%m%d") +\
                         "/" + _pt.strftime("%Y%m%d") +\
                         "." + _pt.strftime("%H%M") + ".npy"
            self.load_npy_file(_ct, _bslnfn, "baseline")
            _ct += datetime.timedelta(hours=self.timeInterval)

            

    def load_npy_file(self, currDate, fName, fType):
        """
        Load a correponding TEC file into the dict
        """
        if fType == "pred":
            if self.useMask:
                self.tecModelDict[currDate] = numpy.load(fName) * self.maskMat
            else:
                self.tecModelDict[currDate] = numpy.load(fName)
        elif fType == "true":
            if self.useMask:
                self.tecTrueDict[currDate] = numpy.load(fName) * self.maskMat
            else:
                self.tecTrueDict[currDate] = numpy.load(fName)
        else:
            if self.useMask:
                self.tecBaselineDict[currDate] = numpy.load(fName) * self.maskMat
            else:
                self.tecBaselineDict[currDate] = numpy.load(fName)

    def plot_data(self, figName, pltType="pred", nCols=3,\
             cmap="jet",\
                    refInpDir="/sd-data/med_filt_tec/",\
                    refFileDate=datetime.datetime(2015,1,1)):
        """
        Plot the data!
        pltType : 1) pred - only predicted TEC maps
                  2) true - only true TEC maps
        nCols : number of columns in the plot grid.
        """
        # read in the data
        if pltType == "pred":
            self.load_pred_tec_data()
            tecDictKeys = self.tecModelDict.keys()
        else:
            self.load_true_tec_data()
            tecDictKeys = self.tecTrueDict.keys()
        # We need to convert the TEC array to a DF for plotting
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
        # Now we'll have to decide on the plot grid
        # we'll try to have a set num of columns 
        # and decide the no.of rows accordingly!
        nRows = len( tecDictKeys )/nCols
        # Now set the plot
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(nRows, nCols, sharex='col', sharey='row')
        # plot the data
        pltCntr = 0
        for i in range(nRows):
            for j in range(nCols):
                if pltType == "pred":
                    # replace the values of dfRef with the tec data
                    _tk = tecDictKeys[pltCntr]
                    dfRef[dfRef.columns] = self.tecModelDict[_tk]
                    # unpivot the DF
                    tecDF = dfRef.unstack().reset_index(name='med_tec')
                    pltDF = tecDF[ ["Mlon", "Mlat",\
                        "med_tec"] ].pivot( "Mlon", "Mlat" )
                    pltDF = pltDF.fillna(0.)
                    mlonVals = pltDF.index.values
                    mlatVals = pltDF.columns.levels[1].values
                    mlonCntr, mlatCntr  = numpy.meshgrid( mlonVals, mlatVals )
                    # Mask the nan values! pcolormesh can't handle them well!
                    cntVals = numpy.ma.masked_where(\
                                    pltDF["med_tec"].values == 0,\
                                    pltDF["med_tec"].values)
                    tecPlot = ax[i, j].pcolormesh(mlonCntr.T , mlatCntr.T, cntVals,\
                                    cmap=cmap, vmin=0, vmax=20)
                    titleStr = _tk.strftime("%Y%m%d-%H%M") + "UT"
                    ax[i, j].set_title(titleStr, fontsize=8)
                else:
                    # replace the values of dfRef with the tec data
                    _tk = tecDictKeys[pltCntr]
                    dfRef[dfRef.columns] = self.tecTrueDict[_tk]
                    # unpivot the DF
                    tecDF = dfRef.unstack().reset_index(name='med_tec')
                    pltDF = tecDF[ ["Mlon", "Mlat",\
                        "med_tec"] ].pivot( "Mlon", "Mlat" )
                    pltDF = pltDF.fillna(0.)
                    mlonVals = pltDF.index.values
                    mlatVals = pltDF.columns.levels[1].values
                    mlonCntr, mlatCntr  = numpy.meshgrid( mlonVals, mlatVals )
                    # Mask the nan values! pcolormesh can't handle them well!
                    cntVals = numpy.ma.masked_where(\
                                    pltDF["med_tec"].values == 0,\
                                    pltDF["med_tec"].values)
                    tecPlot = ax[i, j].pcolormesh(mlonCntr.T , mlatCntr.T, cntVals,\
                                    cmap=cmap, vmin=0, vmax=20)
                    titleStr = _tk.strftime("%Y%m%d-%H%M") + "UT"
                    ax[i, j].set_title(titleStr, fontsize=8)
                pltCntr += 1

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(tecPlot, cax=cbar_ax)
        cbar.set_label('TEC Units')
        fig.savefig(figName, bbox_inches='tight')
        
    def plot_cmpr_mod(self, figName, cmap="jet",\
                    refInpDir="/sd-data/med_filt_tec/",\
                    refFileDate=datetime.datetime(2015,1,1)):
        """
        Plot the data!
        nCols : number of columns in the plot grid.
        """
        # read in the data
        self.load_pred_tec_data()
        self.load_true_tec_data()
        self.load_baseline_tec_data()
        tecDictKeys = self.tecModelDict.keys()
        # We need to convert the TEC array to a DF for plotting
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
        # Now we'll have to decide on the plot grid
        # we'll try to have a set num of columns 
        # and decide the no.of rows accordingly!
        nCols = 3
        nRows = len( tecDictKeys )
        # Now set the plot
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(nRows, nCols, sharex='col', sharey='row')
        # plot the data
        for _r in range(nRows):
            _tk = tecDictKeys[_r]
            # plot model pred
            dfRef[dfRef.columns] = self.tecModelDict[_tk]
            # unpivot the DF
            tecDF = dfRef.unstack().reset_index(name='med_tec')
            pltDF = tecDF[ ["Mlon", "Mlat",\
                "med_tec"] ].pivot( "Mlon", "Mlat" )
            pltDF = pltDF.fillna(0.)
            mlonVals = pltDF.index.values
            mlatVals = pltDF.columns.levels[1].values
            mlonCntr, mlatCntr  = numpy.meshgrid( mlonVals, mlatVals )
            # Mask the nan values! pcolormesh can't handle them well!
            cntVals = numpy.ma.masked_where(\
                            pltDF["med_tec"].values == 0,\
                            pltDF["med_tec"].values)
            tecPlot = ax[_r, 0].pcolormesh(mlonCntr.T , mlatCntr.T, cntVals,\
                            cmap=cmap, vmin=0, vmax=20)
            if _r == 0:
                ax[_r, 0].set_title("STResNet", fontsize=8)
            # plot true tec
            dfRef[dfRef.columns] = self.tecTrueDict[_tk]
            # unpivot the DF
            tecDF = dfRef.unstack().reset_index(name='med_tec')
            pltDF = tecDF[ ["Mlon", "Mlat",\
                "med_tec"] ].pivot( "Mlon", "Mlat" )
            pltDF = pltDF.fillna(0.)
            mlonVals = pltDF.index.values
            mlatVals = pltDF.columns.levels[1].values
            mlonCntr, mlatCntr  = numpy.meshgrid( mlonVals, mlatVals )
            # Mask the nan values! pcolormesh can't handle them well!
            cntVals = numpy.ma.masked_where(\
                            pltDF["med_tec"].values == 0,\
                            pltDF["med_tec"].values)
            tecPlot = ax[_r, 1].pcolormesh(mlonCntr.T , mlatCntr.T, cntVals,\
                            cmap=cmap, vmin=0, vmax=20)
            if _r == 0:
                titleStr = _tk.strftime("%Y%m%d-%H%M") + "UT (TRUE)"
                ax[_r, 1].set_title(titleStr, fontsize=8)
            else:
                titleStr = _tk.strftime("%Y%m%d-%H%M") + "UT"
                ax[_r, 1].set_title(titleStr, fontsize=6)
            # plot baseline pred
            dfRef[dfRef.columns] = self.tecBaselineDict[_tk]
            # unpivot the DF
            tecDF = dfRef.unstack().reset_index(name='med_tec')
            pltDF = tecDF[ ["Mlon", "Mlat",\
                "med_tec"] ].pivot( "Mlon", "Mlat" )
            pltDF = pltDF.fillna(0.)
            mlonVals = pltDF.index.values
            mlatVals = pltDF.columns.levels[1].values
            mlonCntr, mlatCntr  = numpy.meshgrid( mlonVals, mlatVals )
            # Mask the nan values! pcolormesh can't handle them well!
            cntVals = numpy.ma.masked_where(\
                            pltDF["med_tec"].values == 0,\
                            pltDF["med_tec"].values)
            tecPlot = ax[_r, 2].pcolormesh(mlonCntr.T , mlatCntr.T, cntVals,\
                            cmap=cmap, vmin=0, vmax=20)
            if _r == 0:
                ax[_r, 2].set_title("Baseline", fontsize=8)

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(tecPlot, cax=cbar_ax)
        cbar.set_label('TEC Units')
        fig.savefig(figName, bbox_inches='tight')

if __name__ == "__main__":
    modelName = "model_batch64_epoch100_resnet100_nresfltr12_nfltr12_of2_otec12_cf2_csl72_pf12_psl72_tf36_tsl8_gs32_ks55_exoT_nrmT_w0_yr_11_13_379.3419065475464_values"
    baseModelDir = "/sd-data/DeepPredTEC/ModelValidation/"
    timeRange = [ datetime.datetime(2015,3,5), datetime.datetime(2015,3,6) ]
    timeInt = 2 # hours
    dpObj = DatePlots(baseModelDir, modelName, timeRange, timeInt)
    figName = "/home/bharat/Desktop/marc-examples/t/true-plots.pdf"
    dpObj.plot_data(figName, pltType="true", nCols=4)