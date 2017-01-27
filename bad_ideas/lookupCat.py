
import pandas as pd
import re

def lookupTransactions(transData):
    """
    categorize transactions for major retailers from a lookup table
    input transData: dataframe containing cleaned transaction data, 
                     with a merchant column
    updates transData with a 'category' column, containing
        NaN: unknown
        0: food
        1: transport
        2: retail
        TODO standardize this from a file
    """
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
    
    # TODO janky and not generalizable
    transData['category'] = None
    transData['category'] = 1*transData.isFood + 2*transData.isTransport + 3*transData.isRetail - 1
    ll = transData.columns.get_loc('isFood')
    ul = ll+4
    transData.drop(transData.columns[ll:ul], axis=1, inplace=True)



    