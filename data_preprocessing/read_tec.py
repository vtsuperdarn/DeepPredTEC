import pandas as pd
import numpy as np
import sqlite3
import datetime as dt
import os

def generate_tec_map_files(sdate, edate=None, mlat_min=15.,
                           mlon_west=250, mlon_east=34.,
                           inpDir="/sd-data/med_filt_tec/",
                           outDir="../data/tec_map/original/"):

    """Generates 2D TEC map and stors them as files
    """
    # Calc tec map dimension
    tec_map_dim = (int(90-mlat_min), int(((360-mlon_west%360) + mlon_east)/2.)+1)

    # Read the median filtered TEC data
    inpColList = [ "dateStr", "timeStr", "Mlat",\
                   "Mlon", "med_tec", "dlat", "dlon" ]

    if edate is None:
        edate = sdate
    cdates = [sdate + dt.timedelta(days=i) for i in range((edate-sdate).days + 1)]
    for cdate in cdates:
        # Create a folder for storing a day of data files
        rel_dir = cdate.strftime("%Y%m%d") + "/"
        file_dir = outDir + rel_dir
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        inpFile = inpDir + "tec-medFilt-" + cdate.strftime("%Y%m%d") + ".txt"
        print("reading data for " + cdate.strftime("%m/%d/%Y"))
        df = pd.read_csv(inpFile, delim_whitespace=True,
                         header=None, names=inpColList)

        # Change Mlon range from 0 to 360 to -180 to 180
        df.loc[:, "Mlon"] = df.Mlon.apply(lambda x: x if x<=180 else x-360)
        mlon_west = mlon_west - 360

        dlat = df.dlat.iloc[0]
        dlon = df.dlon.iloc[0]
        all_mlats = set(np.arange(mlat_min, 90., dlat))
        all_mlons = set(np.arange(mlon_west, mlon_east+1., dlon))
        if not df.empty:
            dateStr = str(df.dateStr[0])
            # Loop through each 5-min frame
            for time, group in df.groupby("timeStr"):

                # Construct a datetime for current frame
                timeStr = "00" + str(time)
                dtmStr = dateStr + timeStr[-4:]
                dtm = dt.datetime.strptime(dtmStr, "%Y%m%d%H%M")

                # Extract a 2D matrix of interest
                grb = group.loc[(group.Mlat >= mlat_min) & (group.Mlon >= mlon_west) & (group.Mlon <= mlon_east)]
                tec_map = grb.pivot(index="Mlat", columns="Mlon", values="med_tec").as_matrix()
                
                # Check tec_map dimension.
                # If there are missing Mlat or Mlon, insert them with their tec values set to NaN
                if tec_map.shape != tec_map_dim:
                    mlats_exist = set(grb.Mlat.unique())
                    mlons_exist = set(grb.Mlon.unique())
                    mlats_missing = list(all_mlats - mlats_exist)
                    mlons_missing = list(all_mlons - mlons_exist)
                    grb_tmp = grb.copy(deep=True)
                    if mlats_missing:
                        for lat in mlats_missing:
                            grb_tmp.loc[grb_tmp.index[-1]+1, ["Mlat", "Mlon"]] = [lat, grb_tmp.Mlon.iloc[0]]
                    if mlons_missing:
                        for lon in mlons_missing:
                            grb_tmp.loc[grb_tmp.index[-1]+1, ["Mlat", "Mlon"]] = [grb_tmp.Mlat.iloc[0], lon]
                    tec_map = grb_tmp.pivot(index="Mlat", columns="Mlon", values="med_tec").as_matrix()

                file_name = file_dir + dtm.strftime("%Y%m%d.%H%M") + ".npy"
                #np.save(file_name, tec_map)
                with open(file_name, "w") as f:
                    np.save(f, tec_map)

    return

def fill_tec_map(sdate, edate=None, 
                 inpDir="../data/tec_map/original/",
                 outDir="../data/tec_map/filled/"):
    """Fills the missing data in each TEC map. Filled maps are stored as files"""

    if edate is None:
        edate = sdate

    # Create a dataframe
    edate = edate + dt.timedelta(days=1)    # Make the end date inclusive
    nmaps = int(round((edate - sdate).total_seconds() / 60. / tec_resolution))
    dtms = [sdate + dt.timedelta(minutes=tec_resolution*i) for i in range(nmaps)] 

    # Loop through each file
    for dtm in dtms:
        file_name = inpDir + dtm.strftime("%Y%m%d") + "/" +\
                    dtm.strftime("%Y%m%d.%H%M") + ".npy"
        if os.path.isfile(file_name):
            tec_map = np.load(file_name)

            # Fill the missing values
            # NOTE: This step has to be done more properly 
            tec_map[np.isnan(tec_map)] = -1

            file_name_new = outDir + dtm.strftime("%Y%m%d") + "/" +\
                            dtm.strftime("%Y%m%d.%H%M") + ".npy"
            np.save(file_name_new, tec_map)
        else:
            continue
    return

def create_tec_map_table(sdate, edate, tec_resolution=5,
                         file_dir="../data/tec_map/filled/",
                         table_name="tec_map_filled", 
                         db_name="tec_map.sqlite", 
                         db_dir="../data/sqlite3/"):
    """Creats a table in SQLite db to store datetimes of tec maps
       and their file paths"""

    # Make a db connection
    conn = sqlite3.connect(db_dir + db_name)

    # Create a table
    schema = "Create Table IF NOT EXISTS {tbl} (" +\
	     "datetime TIMESTAMP, "+\
             "file_path TEXT, " +\
             "PRIMARY KEY datetime)"
    schema = schema.format(tbl=table_name)

    # Create a dataframe
    edate = edate + dt.timedelta(days=1)    # Make the end date inclusive
    nmaps = int(round((edate - sdate).total_seconds() / 60. / tec_resolution))
    dtms = [sdate + dt.timedelta(minutes=tec_resolution*i) for i in range(nmaps)] 
    files_all = [file_dir + dtm.strftime("%Y%m%d") + "/" +\
                 dtm.strftime("%Y%m%d.%H%M") + ".npy" for dtm in dtms]
    files = [f if os.path.isfile(f) else "NaN" for f in files_all]
    df = pd.DataFrame(data={"datetime":dtms, "file_path":files})

    # Write data to db
    df.to_sql(table_name, conn, schema=schema, if_exists="append", index=False)

    return


if __name__ == "__main__":


    # initialize parameters
    sdate = dt.datetime(2015, 1, 1)
    edate = dt.datetime(2015, 2, 1)

    tec_resolution = 5

    inpDir = "/sd-data/med_filt_tec/"
    outDir="../data/tec_map/original/"
    mlat_min = 15.
    mlon_west = 250
    mlon_east = 34
    generate_tec_map_files(sdate, edate=edate, 
                           mlat_min=mlat_min, mlon_west=mlon_west,
                           mlon_east=mlon_east,
                           inpDir=inpDir, outDir=outDir)

    fill_tec_map(sdate, edate=edate, 
                 inpDir="../data/tec_map/original/",
                 outDir="../data/tec_map/filled/")


#    # Create a table for storing tec map datetimes and file paths
#    create_tec_map_table(sdate, edate, tec_resolution=tec_resolution,
#                         table_name="tec_map", 
#                         db_name="tec_map.sqlite", 
#                         db_dir="../data/sqlite3/")

#
#    #closeness is sampled 12 times every 5 mins, lookback = (12*5min = 1 hour)
#    #freq 1 is 5mins
#    closeness_freq = 1
#    #size corresponds to the sample size
#    closeness_size = 12
#    #period is sampled 24 times every 1 hour (every 12th index), lookback = (24*12*5min = 1440min = 1day)
#    period_freq = 12
#    period_size = 24
#    #trend is sampled 24 times every 3 hours (every 36th index), lookback = (8*36*5min = 1440min = 1day)
#    trend_freq = 36
#    trend_size = 8

#    # Plot a TEC map
#    import matplotlib.pyplot as plt
#    fig, ax = plt.subplots()
#    ax.pcolormesh(data_dict[data_dict.keys()[0]])
#    plt.show()
    

