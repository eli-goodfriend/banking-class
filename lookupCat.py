"""
categorize transactions for major retailers from a lookup table
this will be a training set for learning to categorize transactions generally
"""
import pandas as pd
import re

# TODO should move to SQL
datafile = '/home/eli/Data/Narmi/cleaned_big.csv'
transData = pd.read_csv(datafile,encoding='latin-1')

# TODO janky
foodMerchants = ['pizza','safeway','food','grocer','cafe','chipotle','mc[.]donalds','deli']
transportMerchants = ['lyft','uber','greyhound']
retailMerchants = ['target','walmart','amazon']
# next step healthMerchants = ['pharmacy','gym']
# next step entertainmentMerchants = ['cinema','theater','theatre']

# TODO this is also janky
regex = '|'.join(foodMerchants)
transData['isFood'] = transData['merchant'].str.contains(regex,flags = re.IGNORECASE)

regex = '|'.join(transportMerchants)
transData['isTransport'] = transData['merchant'].str.contains(regex,flags = re.IGNORECASE)

regex = '|'.join(retailMerchants)
transData['isRetail'] = transData['merchant'].str.contains(regex,flags = re.IGNORECASE)

transData['isUnknown'] = ~(transData.isFood | transData.isTransport | transData.isRetail)

transData.to_csv('categorized.csv')