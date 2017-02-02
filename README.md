Transact is a python module to parse and categorize banking transaction data. It is designed for use within a bank's existing data pipeline to analyze transactions as they come from the merchant, before they are passed to the consumer's statement.

Currently, Transact takes csv files of transactions and outputs a csv file of parsed and categorized transactions.

## How to use

python transact.py prayer

## Overview of algorithm

First, the data must be cleaned and parsed. Banking transaction data is heterogeneous, with different merchants and debit card companies providing different information. This information is also often abridged due to character count restrictions. The type of transaction, date and time, merchant, and location are also all jumbled in a single column, and must be extracted.

```
Branch Cash Withdrawal 12/08 17:11:05 POS SHANGHAI RESTAURANTWASHINGTON   DC
```
becomes
```
SHANGHAI RESTAURANT
```

Once the merchant data has been extracted and cleaned, it is categorized. This is currently done using the similarity between words in the merchant description and the desired categories. Remaining transactions are categorized using a model that includes the time and dollar amount of the purchase. 

```
SHANGHAI RESTAURANT -- food
```

[cartoon of the process]

## Details: parsing
All parsing is done using regular expressions[link to code], with a supplementary text file of US cities and states [link to file].

While more sophisticated parsing could be added--e.g. using the parserator package[link]--it is not clear whether these methods would return substantially better results than regular expressions.  Parserator works by conditional random fields[link], yadda yadda yadda.  Banking transaction data does not have reliable structure for the parser to learn.  

## Details: categorization
Categorization is primarily accomplished by comparing the similarity of words in the merchant description to words in the category names.  First, the words are converted into vectors using a word embedding trained on [Common Crawl](commoncrawl.org) data using the [GloVe algorithm](nlp.stanford.edu/projects/glove).  The similarity is then measured as the cosine distance between the vectors.

I chose to use a pre-trained embedding rather than training my own, using a model like [GloVe](nlp.stanford.edu/projects/glove) or [the gensim implementation of word2vec](https://radimrehurek.com/gensim/models/word2vec.html).  Training such a model would be time consuming, not particularly enlightening, and add little value to the accuracy of this project. The best data set for categorizing banking data includes both brand names and general English words, and the web crawl accomplishes exactly that. 

Word embedding models like word2vec and GloVe generally work by yadda yadda yadda.



  
