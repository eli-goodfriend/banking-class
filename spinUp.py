"""
spin up the logistic regression using the training data set
this will create an initial trained logistic regression
future iterations will refine the logistic regression and add new (words) features
the bulk of the operation is reading new transactions and refining the logreg
"""
import transact as ts
from initial_setup import directories as dirs

modelname = dirs.run_dir + 'model_data/transaction_logreg'
filein = dirs.data_dir + 'train_cat.csv'
fileout = dirs.data_dir + 'train_cat.csv'

# running the parser takes most of the time right now, so option to shut it off
ts.run_cat(filein,modelname,fileout,new_run=True,run_parse=False)
