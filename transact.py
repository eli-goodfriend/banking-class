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
    # does it end with a country code?
    # TODO country code list, not just US
    trans_data['country'] = trans_data['raw'].str.extract('(US)$', expand = True)
    trans_data['remainder'] = trans_data['remainder'].str.replace('US', '')
    trans_data['remainder'] = trans_data['remainder'].str.strip()
    
    # find the state # TODO untidy
    us_states = state_data['state']
    states = us_states.to_string(header=False, index=False)
    states = re.sub('\n','|',states)
    regex = '(' + states + ')$'
    trans_data['state'] = trans_data['remainder'].str.extract(regex, expand = True)
    trans_data['remainder'] = trans_data['remainder'].str.replace(regex, '')
    #TODO what if the state isn't at the end?    
    trans_data['remainder'] = trans_data['remainder'].str.strip()

    
    # find if any cities in this state are present as a substring
    # if there are, save them under 'city' column
    # TODO misses Dulles airport (probably among others)
    trans_data['city'] = ""
    for index, row in trans_data.iterrows(): # TODO vectorize this
        st = row['state']
        possible_cities = state_data.loc[state_data['state'] == st]
        possible_cities = possible_cities['cities'] # TODO untidy
        if len(possible_cities) > 0: # this is not nan # TODO untidy
            possible_cities = possible_cities.values[0]
            toSearch = row['remainder']
            city = re.findall(r'|'.join(possible_cities), toSearch, re.IGNORECASE)
            if len(city) > 0:
                city = city[0] # TODO what if find >1 city?
                if isinstance(city, basestring): # TODO janky mc janksalot
                    trans_data['city'][index] = city 
                    current = trans_data['remainder'][index]
                    trans_data['remainder'][index] = re.sub(city,"",current,re.IGNORECASE)
    
    
    
    
    
    
    
    
    
    
    
    