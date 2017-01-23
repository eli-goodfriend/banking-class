"""
read in and parse data
"""
from sqlalchemy import create_engine
import psycopg2
import pandas as pd

# connect to database
dbname = 'narmi_db'
username = 'eli'
password = 'eli'
con = None
connect_str = "dbname='%s' user='%s' host='localhost' password='%s'"%(dbname,username,password)
con = psycopg2.connect(connect_str)

# pull table in as df
sql_query = "SELECT * FROM narmi_data;"
narmi_data = pd.read_sql_query(sql_query,con)

"""
use regex to pull out date and time
"""
timeRegex = '([0-9][0-9]:[0-9][0-9]:[0-9][0-9])'
narmi_data['time'] = narmi_data['raw'].str.extract(timeRegex, expand = True)
narmi_data['remainder'] = narmi_data['raw'].str.replace(timeRegex, '')

dateRegex = '([0-1][0-9]/[0-3][0-9])'
narmi_data['date'] = narmi_data['raw'].str.extract(dateRegex, expand = True)
narmi_data['remainder'] = narmi_data['remainder'].str.replace(dateRegex, '')

"""
remove the phrase 'Branch Cash Withdrawal' since it is in every entry
"""
phrase = 'Branch Cash Withdrawal'
narmi_data['remainder'] = narmi_data['remainder'].str.replace(phrase, '')

print narmi_data.head()

# write back new table to sql database
engine = create_engine('postgresql://%s:%s@localhost:5432/%s'%(username,password,dbname))
narmi_data.to_sql("cleaned_data", engine, if_exists='replace')



