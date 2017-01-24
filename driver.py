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

# pull locations table
sql_query = "SELECT * FROM us_cities;"
us_cities = pd.read_sql_query(sql_query,con)

# clean dates, times, phone numbers, and headers
trans.cleanData(narmi_data)

# find locations, if applicable
trans.findLocations(narmi_data, us_cities)

print narmi_data.head()




