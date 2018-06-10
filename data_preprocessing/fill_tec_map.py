import pandas as pd
import numpy as np
import sqlite3
import datetime as dt
import os

def fill_tec_map(sdate, edate=None, tec_resolution=5,
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
    inpDir="../data/tec_map/original/"
    outDir="../data/tec_map/filled/"

    fill_tec_map(sdate, edate=edate, tec_resolution=tec_resolution,
                 inpDir=inpDir, outDir=outDir)

