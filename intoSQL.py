"""
put csv file into a sql database
make sure the database is running: sudo service postgresql start

this is done only once
"""
from sqlalchemy import create_engine, event, MetaData
from sqlalchemy.schema import DDL
from sqlalchemy_utils import database_exists, create_database
import pandas as pd

datafile = '/home/eli/Data/Narmi/transactions_anonymized.csv'

dbname = 'narmi_db'
username = 'eli'
password = 'eli'

engine = create_engine('postgresql://%s:%s@localhost:5432/%s'%(username,password,dbname))

## create a database (if it doesn't exist)
if not database_exists(engine.url):
    create_database(engine.url)
print(database_exists(engine.url))

# read the data
narmi_data = pd.read_csv(datafile)

# clean off empty columns and rename remaining column
del narmi_data['Unnamed: 1']
del narmi_data['Unnamed: 2']
del narmi_data['Unnamed: 3']
narmi_data.columns = ['raw']

# put it in the database
narmi_data.to_sql('narmi_data', engine, if_exists='replace')