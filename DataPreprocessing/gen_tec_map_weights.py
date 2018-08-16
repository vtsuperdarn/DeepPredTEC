import pandas as pd
import numpy as np
import sqlite3
import datetime as dt
import os

def generate_tec_map_weights(nan_replace=1, mlat_min=15.,
                           mlon_west=250, mlon_east=34.,
                           inpDir="/sd-data/med_filt_tec/",
                           outDir="../data/tec_map/original/"):

    """Generates 2D TEC weight for loss calculation
    """
    # First generate the weights dict
    # we'll setup a weight dict
    # By default all weights will be 1
    # Now if we want change weights of 
    # a particular region we'll assign new
    # weight values to the region. There are
    # two seperate keys for mlats and mlons
    # and each will contain a dict as a key.
    # This nested dict will have a tuple indicating
    # the range of values(mlats/mlons) where weights 
    # need to be changed
    weightDict = { "mlats":{ (45,70) : 2, (80,90) : 2 },
                   "mlons":None}
    # we'll read in TEC file data and replace med_tec values
    # with weights, this way we'll be sure to place weights
    # in the right locations
    testDate = dt.datetime(2015,1,1)
    testTimeStr = 1200
    # Read the median filtered TEC data
    inpColList = [ "dateStr", "timeStr", "Mlat",\
                   "Mlon", "med_tec", "dlat", "dlon" ]
    # Calc tec map dimension
    tec_map_dim = (int(90-mlat_min), int(((360-mlon_west%360) + mlon_east)/2.)+1)
    # Make Mlon between -180 to 180
    mlon_west = mlon_west - 360
    # read the tec file data
    inpFile = inpDir + "tec-medFilt-" + testDate.strftime("%Y%m%d") + ".txt"
    df = pd.read_csv(inpFile, delim_whitespace=True,
                     header=None, names=inpColList)
    # Change Mlon range from 0-360 to -180 to 180
    df.loc[:, "Mlon"] = df.Mlon.apply(lambda x: x if x<=180 else x-360)
    df = df[ (df["timeStr"] == testTimeStr) &
        (df["Mlat"] >= mlat_min) &\
        (df["Mlon"] >= mlon_west) &\
        (df["Mlon"] <= mlon_east) ].reset_index(drop=True)
    # reaplce med_filt tec values with weights
    # make all the values as 1
    df["med_tec"] = 1
    # Now see if there are any new weights
    # see if there are any weight changes 
    # in mlats
    if weightDict["mlats"] is not None:
        # loop through the keys and change weights
        for key in weightDict["mlats"].keys():
            df.loc[ (df["Mlat"] >= key[0]) &\
                 (df["Mlat"] <= key[1]), "med_tec" ] = weightDict["mlats"][key]
    else:
        print "no mlat changes"
    if weightDict["mlons"] is not None:
        for key in weightDict["mlons"].keys():
            df.loc[ (df["Mlon"] >= key[0]) &\
                 (df["Mlon"] <= key[1]), "med_tec" ] = weightDict["mlons"][key]
    else:
        print "no mlon changes"    
    
    # create the weights arr
    # also note we replace nan values here
    # these are most likely missing data points
    weightArr = df.pivot(index="Mlat", columns="Mlon",\
                 values="med_tec").fillna(nan_replace).as_matrix()
    # Now we may have many different combinations of weights
    # we'll auto-generate the filenames to seperate them out
    # generate filename from weightDict
    weightFileName = outDir + "weight__"
    if weightDict["mlats"] is not None:
        # loop through the keys and change weights
        for key in weightDict["mlats"].keys():
            weightFileName += "mlat_" + str(key[0]) +\
                            "-" + str(key[1]) + "_" +\
                            str(weightDict["mlats"][key]) + "__"
    else:
        weightFileName += "mlat_None__"
    if weightDict["mlons"] is not None:
        for key in weightDict["mlons"].keys():        
            weightFileName += "mlon_" + str(key[0]) +\
                            "-" + str(key[1]) + "_" +\
                            str(weightDict["mlons"][key]) + "__"
    else:
        weightFileName += "mlon_None"
    weightFileName += "__weights.npy"
    # save the filename
    with open(weightFileName, "w") as f:
        np.save(f, weightArr)

    return

if __name__ == "__main__":

    inpDir = "/sd-data/med_filt_tec/"
    outDir="../data/tec_map/original/"    # Make sure you have this folder
    mlat_min = 15.
    mlon_west = 250
    mlon_east = 34
    generate_tec_map_files(sdate, edate=edate, 
                           mlat_min=mlat_min, mlon_west=mlon_west,
                           mlon_east=mlon_east,
                           inpDir=inpDir, outDir=outDir)


