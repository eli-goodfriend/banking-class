Transact is a Python module to parse and categorize banking transaction data. It is designed for use within a bank's existing data pipeline to analyze transactions as they come from the merchant, before they are passed to the consumer's statement.

Currently, Transact takes csv files of transactions and outputs a csv file of parsed and categorized transactions.

## How to use 

### Initial set up
Set up utilities are located in the `initial_setup` subdirectory.

1. Download external files and set directories
  * Set the data and run directories in `initial_setup/directories.py`
  * Download to the data directory
    * Your banking transactions, as csv
    * [City data from the US Census](http://www.census.gov/geo/maps-data/data/gazetteer2015.html)
    * [Word embeddings from GloVe](http://nlp.stanford.edu/projects/glove/)
2. Hand categorize seed data
  * Run `initial_setup/find_seeds.py`
  * Open `model_data/lookup_table.csv` and categorize those merchants
  * Make sure those categories match the categories in `model_data/cats.txt`
3. Pre-process word embeddings
  * Run `initial_setup/load_pretrained.py`
4. Pre-process US Census city data
  * Run `initial_setup/cities_into_df.py`

### Running tests
Tests and example data are available in the `test` folder.

* Unit tests of individual functions are in `test/unit_tests.py`
  * These tests use data included in `test/test_input.csv`
* Full system tests of various classification models are in `test/gold_standard.py`
  * Require splitting data into test and cv sets with `test/initial_split.py`, then hand categorizing those sets

### Training and running model
Transact assumes your data is in a csv with two columns: raw transaction data and transaction amounts.

A full example of initializing and training the model on mock stream data is available in `test/online.py`.  This example also includes the option to test the precision of the logistic regression model at each stage.


## Overview of algorithm

The data must first be cleaned and parsed. Banking transaction data is heterogeneous, with different merchants and debit card companies providing different information. This information is also often abridged due to character count restrictions. The type of transaction, date and time, merchant, and location are also all jumbled in a single column, and must be extracted.

```
Branch Cash Withdrawal 12/08 17:11:05 POS TOKYO RESTAURANTWASHINGTON   DC
```

becomes

```
TOKYO RESTAURANT
```

Once the merchant data has been extracted and cleaned, it is categorized. About 35% of transactions can be categorized by using a lookup table generated from the 100 most common merchants. Remaining transactions are categorized using a probabilistic categorization model that uses the merchant name and the dollar amount of the purchase. 

```
TOKYO RESTAURANT -- food
```

![cartoon of the process](https://dl.dropboxusercontent.com/u/60385619/Eli_Goodfriend_Week4Demo.png)

## Details: parsing
All parsing is done using regular expressions, with a supplementary text file of US cities and states.

While more sophisticated parsing could be added--e.g. using the [parserator package](https://github.com/datamade/parserator)--it is not clear whether these methods would return substantially better results than regular expressions.  Parserator works by using conditional random fields to probabilistically parse an input string into substrings.  This package has been very successful with parsing addresses and names, but banking transaction data does not have reliable structure for the parser to learn.  

## Details: categorization
Transactions are categorized in two rounds. If the merchant is one of the most common 100 merchants, the transaction is categorized using the hand-labeled category of that merchant.  This initial round saves computing time by diverting well known merchants away from the more expensive machine learning categorization.  In addition, it creates a labeled set of transactions, enabling the use of supervised techniques.

The remaining transactions are categorized using logistic regression, with the transaction amounts and the words in the merchant description as features. 

### Word embedding
First, the words in the merchant name are converted into vectors using a word embedding trained on [Common Crawl](commoncrawl.org) data using the [GloVe algorithm](nlp.stanford.edu/projects/glove).  Individual words are combined by averaging, although the optimal method of combining words is [not known](http://stackoverflow.com/questions/29760935/how-to-get-vector-for-a-sentence-from-the-word2vec-of-tokens-in-sentence).  The similarity is then measured as the cosine distance between the vectors.

I chose to use a pre-trained embedding rather than training my own, using a model like [GloVe](nlp.stanford.edu/projects/glove) or [the gensim implementation of word2vec](https://radimrehurek.com/gensim/models/word2vec.html).  Training such a model would be time consuming, not particularly enlightening, and add little value to the accuracy of this project. The best data set for categorizing banking data includes both brand names and general English words, and the web crawl accomplishes exactly that. 

Word embedding models like word2vec and GloVe generally work by training a shallow neural network on word co-occurrence from a large corpus.

### Online supervised classification
The word embedding is combined with the amount of the transaction to form the input features for an online, supervised classification algorithm. In this context, "online" refers to the use case of the algorithm requiring continuous learning from an input stream of data, rather than learning from a single training session. The classification will continue to improve as new transactions are passed through the classifier.

Transact includes options for three such algorithms: logistic regression, passive-aggressive, and naive Bayes.  Logistic regression with a stochastic gradient descent solver is a basic, well tested, and probabilistic classifier.  Passive-aggressive classification is related to SVM, is deterministic, and is specifically designed for online learning.  Naive Bayes is typically faster than the other algorithms because it uses a closed form solution to its maximum likelihood computation, rather than an iterative solution, but usually has worse performance.

In tests, logistic regression and passive-aggressive performed similarly well, with naive Bayes substantially underperforming.

![table of results](https://dl.dropboxusercontent.com/u/60385619/results.png)

The default options for the classification algorithm were chosen by optimizing for maximum **precision**, weighted by class sizes.  I chose this metric to select the best model for consumers using automatic budgeting software.  In this context, it is preferable to fail to categorize some transactions (requiring the user to categorize them manually) than to predict a category incorrectly (causing the user's budget to not represent their spending accurately).

## Example insights
By knowing the categories of transactions, we can do more interesting analysis.  For example, this group of users seems to have a late night shopping urge.

![graph](https://dl.dropboxusercontent.com/u/60385619/avg_amount_per_hour.png)

As much as this group enjoys eating out, they seem to skip breakfast.

![graph](https://dl.dropboxusercontent.com/u/60385619/trans_per_hour.png)

The code generating these plots is available in the `plotting` subdirectory.

## About me
I am a data scientist and ex-academic based in New York City. My projects include analyzing US census data for predictors of poverty, teaching math to incarcerated students, and writing puzzle hunts.  You can find me on [LinkedIn](https://www.linkedin.com/in/eligoodfriend) and on my [personal website](www.eligoodfriend.com).
