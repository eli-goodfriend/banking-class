"""
load raw data from sql
clean out known data types (dates, times, phone numbers, headers)
find locations, if applicable (currently US state and city)
identify and clean business
"""

from sqlalchemy import create_engine
import psycopg2
import pandas as pd
import transact as trans
import time

filename = '/home/eli/Data/Narmi/cities_by_state.pickle' 
filename_all = '/home/eli/Data/Narmi/all_cities.pickle'

# connect to database
dbname = 'narmi_db'
username = 'eli'
password = 'eli'
con = None
connect_str = "dbname='%s' user='%s' host='localhost' password='%s'"%(dbname,username,password)
con = psycopg2.connect(connect_str)

# pull transaction data
sql_query = "SELECT * FROM narmi_data;"
narmi_data = pd.read_sql_query(sql_query,con)
narmi_data = narmi_data.head(1000)

# pull locations table
us_cities = pd.read_pickle(filename)

start = time.time()
# clean dates, times, phone numbers, and headers
trans.cleanData(narmi_data)

# find locations, if applicable
trans.findLocations(narmi_data, us_cities)

# extract merchant from transaction description
trans.findMerchant(narmi_data)

end = time.time()
time_per_pt = (end - start) / len(narmi_data)
print "Time per data point = " + str(time_per_pt) + " seconds."
narmi_data.to_csv('cleaned.csv')
print narmi_data.head()



