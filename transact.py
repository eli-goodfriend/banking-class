"""
set of functions for cleaning and parsing banking data
TODO more object oriented?
"""
from sqlalchemy import create_engine
import psycopg2
import pandas as pd
import re

def cleanData(raw_data):
    """
    use regex to pull out date and time
    """
    timeRegex = '([0-9][0-9]:[0-9][0-9]:[0-9][0-9])'
    raw_data['time'] = raw_data['raw'].str.extract(timeRegex, expand = True)
    raw_data['remainder'] = raw_data['raw'].str.replace(timeRegex, '')
    
    dateRegex = '([0-1][0-9]/[0-3][0-9])'
    raw_data['date'] = raw_data['raw'].str.extract(dateRegex, expand = True)
    raw_data['remainder'] = raw_data['remainder'].str.replace(dateRegex, '')
    # TODO there are more dates, like Jul 25
    
    """
    remove the phrase 'Branch Cash Withdrawal' since it is in every entry
    """
    phrase = 'Branch Cash Withdrawal'
    raw_data['remainder'] = raw_data['remainder'].str.replace(phrase, '')
    
    """
    pull out any phone numbers
    """
    phoneRegex = '([0-9][0-9][0-9]-[0-9][0-9][0-9]-[0-9][0-9][0-9])'
    raw_data['phone'] = raw_data['raw'].str.extract(phoneRegex, expand = True)
    raw_data['remainder'] = raw_data['remainder'].str.replace(phoneRegex, '')
    
def findLocations(trans_data, state_data):
    # find the state # TODO untidy
    us_states = state_data['state']
    states = us_states.to_string(header=False, index=False)
    states = re.sub('\n','|',states)
    regex = '(' + states + ')$'
    trans_data['state'] = trans_data['remainder'].str.extract(regex, expand = True)
    trans_data['remainder'] = trans_data['remainder'].str.replace(regex, '')
    #TODO what if the state isn't at the end?
    
    
    # find the cities in each state
    # TODO only do this once
    
    
    # find if any cities in this state are present as a substring
    
    
    
    
    
    
    