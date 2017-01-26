"""
load cleaned data from sql
see if can categorize transaction based on merchant name alone
from lookup table
"""

import pandas as pd
import lookupCat as lc
import time

fileClean = '/home/eli/Data/Narmi/train_clean.csv'
fileCat = '/home/eli/Data/Narmi/train_lookupCat.csv'

narmi_data = pd.read_csv(fileClean)

start = time.time()
lc.lookupTransactions(narmi_data)
end = time.time()


time_per_pt = (end - start) / len(narmi_data)
print "Time per data point = " + str(time_per_pt) + " seconds."
narmi_data.to_csv(fileCat)