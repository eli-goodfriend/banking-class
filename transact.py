"""
set of functions for cleaning and parsing banking data
TODO more object oriented?
"""
from sqlalchemy import create_engine
import psycopg2
import pandas as pd
import re

def cleanData(raw_data):
    # use regex to pull out date and time
    timeRegex = '([0-9][0-9]:[0-9][0-9]:[0-9][0-9])'
    raw_data['time'] = raw_data['raw'].str.extract(timeRegex, expand = True)
    raw_data['description'] = raw_data['raw'].str.replace(timeRegex, '')
    
    dateRegex = '([0-1][0-9]/[0-3][0-9])'
    raw_data['date'] = raw_data['raw'].str.extract(dateRegex, expand = True)
    raw_data['description'] = raw_data['description'].str.replace(dateRegex, '')
    # TODO there are more dates, like Jul 25
    # --- but some of those are part of Uber transactions...
    
    # remove the phrase 'Branch Cash Withdrawal' since it is in every entry
    phrase = 'Branch Cash Withdrawal'
    raw_data['description'] = raw_data['description'].str.replace(phrase, '')
    
    # pull out any phone numbers
    phoneRegex = '([0-9][0-9][0-9]-[0-9][0-9][0-9]-[0-9][0-9][0-9])'
    raw_data['phone'] = raw_data['raw'].str.extract(phoneRegex, expand = True)
    raw_data['description'] = raw_data['description'].str.replace(phoneRegex, '')
    
    # remove the POS designation from the front of descriptions
    regex = '^POS '
    raw_data['description'] = raw_data['description'].str.strip()
    raw_data['description'] = raw_data['description'].str.replace(regex, '')
    
    
def findLocations(trans_data, state_data):
    # does it end with a country code?
    # TODO country code list, not just US
    trans_data['country'] = trans_data['raw'].str.extract('(US)$', expand = True)
    trans_data['description'] = trans_data['description'].str.replace('US', '')
    trans_data['description'] = trans_data['description'].str.strip()
    
    # find the state # TODO untidy
    us_states = state_data['state']
    states = us_states.to_string(header=False, index=False)
    states = re.sub('\n','|',states)
    regex = '(' + states + ')$'
    trans_data['state'] = trans_data['description'].str.extract(regex, expand = True)
    trans_data['description'] = trans_data['description'].str.replace(regex, '')
    trans_data['description'] = trans_data['description'].str.strip()
    #TODO what if the state isn't at the end?    

    
    # find if any cities in this state are present as a substring
    # if there are, save them under 'city' column
    # TODO misses Dulles airport (probably among other airport)
    # TODO misses cities that get cut off bc they are too long
    # TODO misses cities that aren't in database bc they are technically neighborhoods
    all_cities = [city for row in state_data['cities'] for city in row]
    trans_data['city'] = ""
    regex = '(' + '|'.join(all_cities) + ')$'
    trans_data['city'] = trans_data['description'].str.extract(regex, re.IGNORECASE)
    trans_data['description'] = trans_data['description'].str.replace(regex, '', case = False, flags = re.IGNORECASE)
    trans_data['description'] = trans_data['description'].str.strip()
    # TODO what if there's another city name in the string for whatever reason?
    # --- what if it finds a city name that's not a city in that state?
    # TODO what if the city isn't at the end?

def findMerchant(trans_data):
    trans_data['merchant'] = trans_data['description']

    # clean out known initial intermediary flags
    # TODO keep this?
    third_parties = ['SQ \*','LEVELUP\*']
    regex = '^(' + '|'.join(third_parties) + ')'
    trans_data['merchant'] = trans_data['merchant'].str.replace(regex, '', case = False, flags = re.IGNORECASE)
    
    
    
    
    
    
    
    
    
    
    