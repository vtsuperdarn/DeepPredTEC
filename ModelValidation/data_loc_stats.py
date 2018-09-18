import datetime
import pandas
import numpy
import dask
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.colors import ListedColormap
from davitpy import utils

class LocDataPnts(object):
    """
    A class to read in data from median filetered TEC files
    and calculate number of data points at different locations!
    """
    def __init__(self, timeRange, inpDir="/sd-data/med_filt_tec/"):
        """
        timeRange : start and end time to calc stats
        inpDir : dir to read TEC data from!
        """
        self.timeRange = timeRange
        self.inpDir = inpDir
        self.tecTrueDict = {}

    def read_data(self, \
                refFileDate=datetime.datetime(2015,1,1)):
        """
        Read data from the csv files
        """
        # depending on the time range and fType get a list
        # of fileNames that need to be loaded!
        # Get a list of all the files in the dir!
        dfList = []
        currTime = self.timeRange[0]
        inpColList = [ "dateStr", "timeStr", "Mlat",\
                   "Mlon", "med_tec", "dlat", "dlon" ]
        mlat_min, mlon_west, mlon_east = 15., -110, 34
        while currTime <= self.timeRange[1]:
            # get the fName for actual data dir, read the data from
            # all the files during the day!
            _cfN = self.inpDir + "tec-medFilt-" +\
                     currTime.strftime("%Y%m%d") + ".txt" 
            mfDF = pandas.read_csv(_cfN, delim_whitespace=True,
                         header=None, names=inpColList)
            # groupby Mlat, Mlon and get the counts
            _locDF = mfDF.groupby(["Mlat", "Mlon"]\
                        ).size().reset_index(name='counts')
            # Change Mlon range from 0-360 to -180 to 180
            _locDF.loc[:, "Mlon"] = _locDF.Mlon.apply(\
                                lambda x: x if x<=180 else x-360)
            _locDF = _locDF[(_locDF.Mlat >= mlat_min) &\
                            (_locDF.Mlon >= mlon_west) &\
                            (_locDF.Mlon <= mlon_east)]
            dfList.append(_locDF)
            currTime += datetime.timedelta(days=1)
        return pandas.concat(dfList)
    
    def generate_stat_plot(self, figName):
        """
        Get the DF size and generate the stat plots
        """
        cntSeaMap = ListedColormap(sns.color_palette("Reds"))
        locDF = self.read_data()
        # Again groupby Mlat, Mlon and get teh total counts
        locStatDF = locDF[["Mlat", "Mlon", "counts"]\
                        ].groupby(["Mlat", "Mlon"]\
                        ).sum().reset_index()
        locStatDF["perc_count"] = locStatDF["counts"]*100.\
                            /locStatDF["counts"].max()
        f = plt.figure(figsize=(12, 8))
        ax = f.add_subplot(1,1,1)
#         m1 = utils.plotUtils.mapObj(boundinglat=10.,\
#                     gridLabels=True, coords="mag", ax=ax, datetime=datetime.datetime(2014,1,1))
        pltCntDF = locStatDF[ ["Mlon", "Mlat",\
                        "perc_count"] ].pivot( "Mlon", "Mlat" )
        pltCntDF = pltCntDF.fillna(0.)
        mlonVals = pltCntDF.index.values
        mlatVals = pltCntDF.columns.levels[1].values
        mlonCntr, mlatCntr  = numpy.meshgrid( mlonVals, mlatVals )
        # Mask the nan values! pcolormesh can't handle them well!
        cntVals = numpy.ma.masked_where(\
                        numpy.isnan(pltCntDF["perc_count"].values),\
                        pltCntDF["perc_count"].values)
#         dataPntPlot = m1.pcolormesh(mlonCntr.T , mlatCntr.T, cntVals,\
#                         cmap=cntSeaMap, zorder=7, latlon=True)
        dataPntPlot = ax.pcolormesh(mlonCntr.T , mlatCntr.T, cntVals,\
                        cmap=cntSeaMap)
        cbar = plt.colorbar(dataPntPlot, orientation='vertical')
        cbar.set_label('Data Coverage')
        ax.set_ylabel("MLAT", fontsize=14)
        ax.set_xlabel("MLON", fontsize=14)
        ax.tick_params(labelsize=14)
        f.savefig(figName, bbox_inches='tight')

    

if __name__ == "__main__":
    timeRange = [ datetime.datetime(2012,1,1), datetime.datetime(2012,12,31) ]
    tsObj = LocDataPnts(timeRange)
    figName = "/home/bharat/Desktop/marc-examples/t/data_cov.pdf"
    tsObj.generate_stat_plot(figName)


