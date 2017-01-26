"""
load raw data from sql
clean out known data types (dates, times, phone numbers, headers)
find locations, if applicable (currently US state and city)
identify and clean business
"""

import pandas as pd
import parse as ps
import time

fileTrans = '/home/eli/Data/Narmi/train.csv'
fileCities = '/home/eli/Data/Narmi/cities_by_state.pickle' 
fileClean = '/home/eli/Data/Narmi/train_clean.csv'

narmi_data = pd.read_csv(fileTrans)
us_cities = pd.read_pickle(fileCities)

start = time.time()
ps.parseTransactions(narmi_data,'raw',us_cities)
end = time.time()


time_per_pt = (end - start) / len(narmi_data)
print "Time per data point = " + str(time_per_pt) + " seconds."
narmi_data.to_csv(fileClean)



