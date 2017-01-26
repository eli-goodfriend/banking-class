"""
parseTransactions regex parser
inputs df: pandas dataframe with unparsed transactions
       col: name of column with unparsed transactions
       cities: pandas dataframe of city names
updates df with new parsed columns       
"""
import re

def throwOut(df,col,regex):
    df[col] = df[col].str.replace(regex,'',case = False,flags = re.IGNORECASE)
    
def move(df,colIn,colOut,regex):
    df[colOut] = df[colIn].str.extract(regex,flags = re.IGNORECASE,expand = True)
    throwOut(df,colIn,regex)
    
def strip(df,col):
    df[col] = df[col].str.strip()
    
def makeDesc(df,col):
    df['description'] = df[col]

def cleanData(df):
    # use regex to pull out date and time
    move(df,'description','time','([0-9][0-9]:[0-9][0-9]:[0-9][0-9])')  
    move(df,'description','date','([0-1][0-9]/[0-3][0-9])')

    # remove the phrase 'Branch Cash Withdrawal' since it is in every entry
    throwOut(df,'description','Branch Cash Withdrawal')
    
    # pull out any phone numbers
    # TODO misses 800-COMCAST
    move(df,'description','phone','([0-9][0-9][0-9]-[0-9][0-9][0-9]-[0-9][0-9][0-9])')
    
    # remove the POS designation from the front of descriptions
    strip(df,'description')
    throwOut(df,'description','^POS ')
    
def findLocations(df, state_data):
    # does it end with a country code?
    # TODO country code list, not just US
    move(df,'description','country','(US)$')
    strip(df,'description')
    
    # find the state # TODO untidy
    us_states = state_data['state']
    states = us_states.to_string(header=False, index=False)
    states = re.sub('\n','|',states)
    regex = '(' + states + ')$'
    move(df,'description','state',regex)
    strip(df,'description')
    # TODO what if the state isn't at the end?    
    # TODO misclassifies BARBRI as BARB located in RI

    
    # find if any cities in this state are present as a substring
    # if there are, save them under 'city' column
    # TODO misses Dulles airport (probably among other airport)
    # TODO misses cities that get cut off bc they are too long
    # TODO misses cities that aren't in database bc they are technically neighborhoods
    all_cities = [city for row in state_data['cities'] for city in row]
    df['city'] = ""
    regex = '(' + '|'.join(all_cities) + ')$'
    move(df,'description','city',regex)
    strip(df,'description')
    # TODO what if there's another city name in the string for whatever reason?
    # --- what if it finds a city name that's not a city in that state?
    # TODO what if the city isn't at the end?

def findMerchant(df):
    df['merchant'] = df['description']

    # clean out known initial intermediary flags
    # TODO keep this in a separate column?
    # TODO get a list of these from somewhere?
    third_parties = ['SQ \*','LEVELUP\*','PAYPAL \*','SQC\*']
    regex = '^(' + '|'.join(third_parties) + ')'
    throwOut(df,'merchant',regex)
    
    # clean out strings that are more than one whitespace unit from the left
    throwOut(df,'merchant','\s\s+.+$')
    strip(df,'merchant')
    
    # clean out the chunks of Xs that come from redacting ID numbers
    throwOut(df,'merchant','X+-?X+')
    strip(df,'merchant')
    
    # clean out strings that look like franchise numbers
    # TODO this is not correct: too broad
    throwOut(df,'merchant','[#]?([0-9]){3,999}$')
    strip(df,'merchant')
    
def parseTransactions(df,col,cities):
    makeDesc(df,col) # initialize the description field

    print 'basic cleaning...'
    cleanData(df) # clean dates, times, phone numbers, and headers

    print 'finding locations...'
    findLocations(df,cities) # find locations, if applicable

    print 'finding merchants...'
    findMerchant(df) # extract merchant from transaction description
    
    
    