import pandas as pd
import numpy as np
import sqlite3
import datetime as dt
import os

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
    files = [f.replace(file_dir, "") if os.path.isfile(f) else "NaN" for f in files_all]
    df = pd.DataFrame(data={"datetime":dtms, "file_path":files})

    # Write data to db
    df.to_sql(table_name, conn, schema=schema, if_exists="append", index=False)

    return

if __name__ == "__main__":

    # initialize parameters
    sdate = dt.datetime(2015, 1, 1)
    edate = dt.datetime(2015, 3, 1)    # Includes the edate
    tec_resolution = 5
    file_dir="../data/tec_map/filled/"

    # Create a table for storing tec map datetimes and file paths
    create_tec_map_table(sdate, edate, tec_resolution=tec_resolution,
                         file_dir=file_dir,
                         table_name="tec_map_filled", 
                         db_name="tec_map.sqlite", 
                         db_dir="../data/sqlite3/")

