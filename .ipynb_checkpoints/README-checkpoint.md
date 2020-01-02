# Twitter Hashtag Miner

## Introduction

Have you ever wondered how you can boost your twitter account and its tweets to reach the maximum possible audience? Well twitter has just created one of the bset ways to categorize tweets and it is through hashtags, originally created by Chris Messina in 2007. This project's aim is to use AI to predict the best possible hashtag for your tweet.

## Navigating the project

`/jupyternb/V{version}_Mining.ipynb` and `/jupyternb/V{version}_Training.ipynb` is for all the ipython notebooks that explains the mining and training process ,respectively.

`/source/v{version}/` Stores the source codes used for the mining and training processes 

`tweetrecommender.py` Run this file with a tweet wrapped in quotes as an argument to predict the most suitable hashtag. 
Ex: 
`python tweetrecommender.py "A wonderful day for soccer"` predicts [\#family]

`/classifier_archive` Stores the final stage classifier (stored as .pickle) of the specific training versions.

Please note that V1's pickled classifier surpassed 3GB and Github does nto support Git files (or even GIT LFS) files larger than 2GB. 
Please download the pickled file from:

https://drive.google.com/file/d/1i-R9r6ksyOnCQPrmmLZw8S7YQtWv8aqW/view?usp=sharing

and store it in classifier_archive as "v1_classifier.pickle" to use the `tweetrecommender.py` for prediction


## Training Versions

### V1

Training Process

- Extract 1000 tweets from each hashtag
- Tokenize the tweet text using an nltk.TreebankTokenizer
- Remove trailing periods
- Replace emojis with their unicode description
- Breakdown compound words (that do not exist in the word embedding vocabulary) into their simpler forms
- Remove the following punctuations: `[".","#",":","•",",","@","\"",";","\'"]`
- Convert each token into a vector using a pretrained word embedding model (glove50)
- Sum up the vectors of tokens for each tweet

### V2

- Tokenizes the tweet using nltk.TweetTokenizer
- Replace emojis with their unicode description
- Looks for other hashtags in the tweet
- Create the label for the tweet consisting of the hashtags
- Remove the following punctuations: `[".","#",":","•",",","@","\"",";","\'"]`
- Traing a neural network to generate word embeddings for each token.
- Sum up the vectors of tokens for each tweet

## Necessary Packages

|Package Name|Version|
|------------|-------|
|pandas|0.25.3|
|scikit-Learn|0.22|
|tweepy|3.8.0|
|numpy|1.17.4|
|nltk|3.4.4|
|emoji|0.5.3|
|gensim|3.8.0|
