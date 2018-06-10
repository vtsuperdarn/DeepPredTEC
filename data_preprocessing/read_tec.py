import pandas as pd
import numpy as np
import datetime as dt

def gen_2d_tec_map(cdate, mlat_min = 15., mlon_west = 250,
                   mlon_east = 34.,
                   inpDir = "/sd-data/med_filt_tec/"):

    """Generates data for 2D TEC map
    """

    # Read the median filtered TEC data
    inpColList = [ "dateStr", "timeStr", "Mlat",\
                   "Mlon", "med_tec", "dlat", "dlon" ]

    inpFile = inpDir + "tec-medFilt-" + cdate.strftime("%Y%m%d") + ".txt"
    print("reading data for " + cdate.strftime("%m/%d/%Y"))
    df = pd.read_csv(inpFile, delim_whitespace=True,
		     header=None, names=inpColList)

    # Change Mlon range from 0 to 360 to -180 to 180
    df.loc[:, "Mlon"] = df.Mlon.apply(lambda x: x if x<=180 else x-360)
    mlon_west = mlon_west - 360

    data_dict = {}
    if not df.empty:
	dateStr = str(df.dateStr[0])
        # Loop through each 5-min frame
	for time, group in df.groupby("timeStr"):

	    # Construct a datetime for current frame
	    timeStr = "00" + str(time)
	    dtmStr = dateStr + timeStr[-4:]
	    dtm = dt.datetime.strptime(dtmStr, "%Y%m%d%H%M")

            grb = group.loc[(group.Mlat >= mlat_min) & (group.Mlon >= mlon_west) & (group.Mlon <= mlon_east)]
            tec_map = grb.pivot(index="Mlat", columns="Mlon", values="med_tec").as_matrix()
            data_dict[dtm] = tec_map

            print("Extracted 2D matrix for " + str(dtm))
            #print("There are {nan_frame} np.nan frames for this date".format(nan_frame=np.)

    return data_dict

if __name__ == "__main__":


    # initialize parameters
    #stm = [dt.datetime(2015, 1, 7, 0, 0)]
    #etm = [dt.datetime(2015, 1, 7, 23, 59)]
    #cdates = [stm + dt.timedelta(days=i) for i in range(etm-stm).days]

    inpDir = "/sd-data/med_filt_tec/"
    cdate = dt.datetime(2015, 1, 7)
    mlat_min = 15.
    mlon_west = 250
    mlon_east = 34

    data_dict = gen_2d_tec_map(cdate, mlat_min=mlat_min, mlon_west=mlon_west,
                               mlon_east=mlon_east, inpDir=inpDir)

#    # Plot a TEC map
#    import matplotlib.pyplot as plt
#    fig, ax = plt.subplots()
#    ax.pcolormesh(data_dict[data_dict.keys()[0]])
#    plt.show()
    

