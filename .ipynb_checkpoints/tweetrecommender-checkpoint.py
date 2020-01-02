from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import gensim.downloader as gensimapi
from nltk.tokenize import TreebankWordTokenizer
from emoji import UNICODE_EMOJI
import unicodedata as ud
import functools
import operator
import re
import numpy as np
from wordfreq import word_frequency
import emoji
from sklearn.base import BaseEstimator, TransformerMixin
import copy
import pandas as pd
from sklearn.pipeline import Pipeline
import sys
from numpy import genfromtxt
import pickle

vectorizermodel = gensimapi.load("glove-wiki-gigaword-50")

class HashtagDetailsExtractor(BaseEstimator, TransformerMixin):
    """
    Returns a list of tweets (in text form) from the given hashtag
    
    """
    def __init__(self,tweetnum=100):
        self.tweetnum=tweetnum
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,tag):
        tweet_text_list = []
        for tweet in tweepy.Cursor(api.search,q=tag,tweet_mode="extended").items(self.tweetnum):
            tweet_text_list.append(tweet._json["full_text"].lower())
        
        return tweet_text_list
   
class Tokenizer(BaseEstimator, TransformerMixin):
    """
    Returns tokenized text with the supplied Tokenizer 
    
    """
    def __init__(self,tokenizer_model,lowercase=True):
        self.tokenizer_model = tokenizer_model
        self.lowercase = lowercase
        
    def fit(self,X,y=None):
        return self
    
    def transform(self, X, y=None):
        tokenized_tweets = []
        for tweet in X:    
            tweet_text_tokenized = self.tokenizer_model.tokenize(tweet)
            tokenized_tweets.append(tweet_text_tokenized)
        return tokenized_tweets

class ConvertEmojis(BaseEstimator, TransformerMixin):
    """
    Extracts the description of the emoji and repalces it with the emoji itself
    
    """
    def __init__(self, tokenizer_model):
        self.tokenizer_model = tokenizer_model
    
    def fit(self,X,y=None):
        return self
    
    def hasemoji(self,s):
        em_split_emoji = emoji.get_emoji_regexp().split(s)
        em_split_whitespace = [substr.split() for substr in em_split_emoji]
        em_split = functools.reduce(operator.concat, em_split_whitespace)
        emojiExists = False
        for emojiTest in em_split:
            if(emojiTest in UNICODE_EMOJI):
                emojiExists = True
    
        return emojiExists
    
    def transform(self, X, y=None):
        X_copy = copy.deepcopy(X)
        for i,tweet in enumerate(X):
            shiftindex = 0
            for ii,word_token in enumerate(tweet):
                if(self.hasemoji(word_token)):
                    em_split_emoji = emoji.get_emoji_regexp().split(word_token)
                    em_split_whitespace = [substr.split() for substr in em_split_emoji]
                    em_split = functools.reduce(operator.concat, em_split_whitespace)
                    emoji_detail_tokenized = []
                    for each_emoji in em_split:
                        try:
                            emoji_detail = ud.name(each_emoji)
                            emoji_tokenized = self.tokenizer_model.tokenize(emoji_detail.lower())
                            if(emoji_tokenized[-1]=="selector-16" and emoji[-2]=="variation"):
                                emoji_tokenized.clear()
                            emoji_detail_tokenized.extend(emoji_tokenized)
                        except:
                            pass
                    X_copy[i].pop(ii + shiftindex)
                    X_copy[i] = X_copy[i][:ii + shiftindex] + emoji_detail_tokenized + X_copy[i][ii + shiftindex:]
                    shiftindex += len(emoji_detail_tokenized) - 1
        return X_copy

class RemoveTrailingPeriods(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, X ,y=None):
        return self
    
    def transform(self, X,y=None):
        for i,tweet in enumerate(X):
            indexes_with_period = []
            for ii, word_token in enumerate(tweet):
                while(X[i][ii][-1]=="." and len(X[i][ii]) != 1):
                    X[i][ii] = X[i][ii][:-1]
                
                if(X[i][ii][-1]=="…"):
                    X[i][ii] = X[i][ii][:-1]          
        return X
                
class RemovePunctuation(BaseEstimator, TransformerMixin):
    
    def __init__(self,charsequence):
        self.charsequence = set(charsequence)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_copy = X.copy()
        for i,tweet in enumerate(X):
            
            X_copy[i] = [text for text in tweet if not set([text]).issubset(self.charsequence)]
            
        return X_copy

class BreakupWords(BaseEstimator, TransformerMixin):
    
    def __init__(self,model):
        self.model = model
        
    def fit(self, X, y=None):
        return self
    
    def break_compound_word(self,compound_word):
        possible_words = []
        first_word=""
        for i,xchar in enumerate(compound_word):
            second_word = compound_word[i+1:]
            if(xchar=="-"):
                try:
                    self.model[first_word]
                    self.model[second_word]
                    possible_words.append([first_word,second_word])
                except:
                    pass
                    
            first_word+=xchar
            try:          
                self.model[first_word]
                self.model[second_word]
                possible_words.append([first_word,second_word])
            except:
                if(second_word==""):
                    try:
                        self.model[first_word]
                        possible_words.append([first_word])
                    except:
                        pass
        return possible_words
    
    def transform(self, X, y=None):
        X_copy = copy.deepcopy(X)
        for i,tweet in enumerate(X):
            shiftindex = 0
            for ii, token in enumerate(tweet):
                child_words = self.break_compound_word(token)
                highestfrq = 0
                if(len(child_words)!= 0):
                    for child_word_set in child_words:
                        if(len(child_word_set)!=1):
                            both_word_freq = word_frequency(child_word_set[0],"en")*word_frequency(child_word_set[1],"en")
                            if (both_word_freq > highestfrq):
                                most_likely_combo = child_word_set
                                highestfrq = both_word_freq
                        else:
                            most_likely_combo = child_word_set
                    X_copy[i].pop(ii + shiftindex)
                    X_copy[i] = X_copy[i][:ii + shiftindex] + most_likely_combo + X_copy[i][ii + shiftindex:]
                    shiftindex += len(most_likely_combo)-1
        return X_copy
                    
class VectorizeTweets(BaseEstimator, TransformerMixin):
    
    def __init__(self,model):
        self.model = model
        
    def fit(self,X,y=None):
        return self
    
    def transform(self, X, y=None):
        X_vectorized = []
        for tweet in X:
            tweet_vectorized = []
            for token in tweet:
                try:
                    vector = self.model[token]
                    tweet_vectorized.append(vector)
                except:
                    pass
            X_vectorized.append(tweet_vectorized)
        
        return X_vectorized
                    
class SumUpTweetVector(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass
    
    def fit(self,X,y=None):
        return self
    
    def transform(self, X, y=None):
        tweet_vector_sums = []
        vector_shape = len(X[0][0])
        for tweet in X:
            vector_sum = np.zeros(vector_shape)
            for vector in tweet:
                vector_sum += vector
                
            tweet_vector_sums.append(vector_sum)
            
        return np.array(tweet_vector_sums)

tokenizermodel = TreebankWordTokenizer()

predictor_path = r"classifier_archive/v1_classifier.pickle"

with open(predictor_path, "rb") as input_file:
    predictor = pickle.load(input_file)


twitter_miner_pipeline = Pipeline(
    [
        ("tokenizer", Tokenizer(tokenizer_model=tokenizermodel)),
        ("trailing_p",RemoveTrailingPeriods()),
        ("break_down_emoji",ConvertEmojis(tokenizer_model=tokenizermodel)),
        ("rm_punctuation",RemovePunctuation([".","#",":","•",",","@","\"",";","\'"])),
        ("breakdown_words",BreakupWords(model=vectorizermodel)),
        ("vectorizer",VectorizeTweets(model=vectorizermodel)),
        ("tweet_vector_sum",SumUpTweetVector()),
        ("predict",predictor),
    ]
)

tweet_text = sys.argv[1]

try:
    hashtag = twitter_miner_pipeline.predict([tweet_text])
except:
    print("Pickled classifier not found. Please view classifier_archive/v1_classifier_temp.txt for further information")

print(hashtag)