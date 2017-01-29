"""
spin up the logistic regression using the training data set
this will create an initial trained logistic regression
future iterations will refine the logistic regression and add new (words) features
the bulk of the operation is reading new transactions and refining the logreg
"""
import transact as ts

modelname = 'transaction_logreg'
filein = '/home/eli/Data/Narmi/train_cat.csv'
fileout = '/home/eli/Data/Narmi/train_cat.csv'

# running the parser takes most of the time right now, so option to shut it off
ts.run_cat(filein,modelname,fileout,new_run=True,run_parse=False)
