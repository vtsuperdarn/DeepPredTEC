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

def fill_missing_tec_map(tec_resolution=5, nframes_before=3, nframes_after=3,
                         table_name="tec_map_filled", 
                         db_name="tec_map.sqlite", 
                         db_dir="../data/sqlite3/",
                         file_dir="../data/tec_map/filled/"):

    """Fills the missing TEC frames"""
    
    # Make a db connection
    conn = sqlite3.connect(db_dir + db_name, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = conn.cursor()

    # Get the missing frames
    command = "SELECT datetime FROM {tbl} "+\
              "WHERE file_path = 'NaN' ORDER BY datetime"
    command = command.format(tbl=table_name)
    cur.execute(command)
    rows = cur.fetchall()
    dtms = [x[0] for x in rows]
    print("There are {nframes} frames missing".format(nframes=len(dtms)))

    # Fill the missing frames one at a time
    for i, dtm in enumerate(dtms):
        stm = dtm - dt.timedelta(minutes=tec_resolution * nframes_before)
        etm = dtm + dt.timedelta(minutes=tec_resolution * nframes_after)

        # Get the frames adjacent to current frame
        command = "SELECT datetime, file_path FROM {tbl} "+\
                  "WHERE datetime BETWEEN '{stm}' AND '{etm}' "+\
                  "ORDER BY datetime"
        command = command.format(tbl=table_name, stm=stm, etm=etm)
        cur.execute(command)
        rows_i = cur.fetchall()
        dtms_i = [x[0] for x in rows_i]
        indx_i = dtms_i.index(dtm) 
        fnames_before = [x[1] for x in rows_i[:indx_i]]
        fnames_after = [x[1] for x in rows_i[indx_i+1:]]

        # Reorder the frames and their datetimes excluding the current frame
        fnames_tmp = fnames_before[::-1] + fnames_after
        for f in fnames_tmp:
            if f != "NaN":
                f_path = file_dir + f 
                tec_map = np.load(f_path)

                # Create a file for current TEC Frame
                rel_dir = dtm.strftime("%Y%m%d") + "/"
                fname_i = file_dir + rel_dir + \
                          dtm.strftime("%Y%m%d.%H%M") + ".npy"
                with open(fname_i, "w") as f_i:
                    np.save(f_i, tec_map)

                # Update the file_path of current TEC map in the SQLite table
                command = "UPDATE {tbl} SET "+\
                          "file_path = '{fname}' "+\
                          "WHERE datetime = '{dtm}'"
                command = command.format(tbl=table_name, fname=fname_i.replace(file_dir, ""),
                                         dtm = dtm)
                cur.execute(command)
                print("Filled the missing frame for " + str(dtm))
            else:
                continue

    # Make a commit
    conn.commit()

    # Close db connection
    conn.close()

    return


if __name__ == "__main__":


    # initialize parameters
    sdate = dt.datetime(2015, 1, 1)
    edate = dt.datetime(2015, 1, 17)    # Includes the edate
    tec_resolution = 5
    file_dir="../data/tec_map/filled/"

#    # Create a table for storing tec map datetimes and file paths
#    create_tec_map_table(sdate, edate, tec_resolution=tec_resolution,
#                         file_dir=file_dir,
#                         table_name="tec_map_filled", 
#                         db_name="tec_map.sqlite", 
#                         db_dir="../data/sqlite3/")

    # Fill the missing frames with adjacent frames
    fill_missing_tec_map(tec_resolution=tec_resolution,
                         nframes_before=3, nframes_after=3,
                         table_name="tec_map_filled", 
                         db_name="tec_map.sqlite", 
                         db_dir="../data/sqlite3/",
                         file_dir=file_dir)

