import pandas as pd
import numpy as np
import sqlite3
import datetime as dt
import os

def fill_tec_map(sdate, edate=None, tec_resolution=5,
                 inpDir="../data/tec_map/original/",
                 outDir="../data/tec_map/filled/"):
    """Fills the missing data in each TEC map.
    Filled maps are stored as seperate files"""

    if edate is None:
        edate = sdate
    cdates = [sdate + dt.timedelta(days=i) for i in range((edate-sdate).days + 1)]
    for cdate in cdates:
        # Create a folder for storing a day of data files
        rel_dir = cdate.strftime("%Y%m%d") + "/"
        file_dir = outDir + rel_dir
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

    # Create a dataframe
    edate = edate + dt.timedelta(days=1)    # Make the end date inclusive
    nmaps = int(round((edate - sdate).total_seconds() / 60. / tec_resolution))
    dtms = [sdate + dt.timedelta(minutes=tec_resolution*i) for i in range(nmaps)] 

    # Loop through each file
    for dtm in dtms:
        print(("filling TEC map for " + str(dtm)))
        file_name = inpDir + dtm.strftime("%Y%m%d") + "/" +\
                    dtm.strftime("%Y%m%d.%H%M") + ".npy"
        if os.path.isfile(file_name):
            tec_map = np.load(file_name)

            # Fill the missing values
            # NOTE: This step needs to be explored more
            tec_map[np.isnan(tec_map)] = -1

            file_name_new = outDir + dtm.strftime("%Y%m%d") + "/" +\
                            dtm.strftime("%Y%m%d.%H%M") + ".npy"
            np.save(file_name_new, tec_map)
        else:
            continue
    return

if __name__ == "__main__":


    # initialize parameters
    sdate = dt.datetime(2011, 1, 1)
    edate = dt.datetime(2016, 12, 31)
    tec_resolution = 5
    inpDir="/home/sd-guest/Documents/data/dask/npy_dask/"#"../data/tec_map/original/"       # Make sure you have this folder
    outDir="/home/sd-guest/Documents/data/tec_filled/"#"../data/tec_map/filled/"         # Make sure you have this folder

    fill_tec_map(sdate, edate=edate, tec_resolution=tec_resolution,
                 inpDir=inpDir, outDir=outDir)

