import sys
import nltk
import re

def reg_alphabetic(s):
    regex = re.compile('[^a-zA-Z ]')
    s = regex.sub('', s)
    return s

def getStopWdSet():
    from nltk.corpus import stopwords
    stopwd = set(stopwords.words('english'))
    return stopwd

def getLemmatizer():
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    return lemmatizer

def tokenization(s):
    return nltk.word_tokenize(s)

def ngramStrExtract(tokens,n):
    l = len(tokens)-n+1
    ngrams = []
    for i in range(l):
        s = tokens[i]
        for j in range(n-1): s+=' '+tokens[i+j+1]
        ngrams.append(s)
    return ngrams

"""
accumulate the ngram occurence
return the dictionary ngram->ngram count
"""
def ngram_stats(ngrams):
    ngrams_statistics = {}
    for ngram in ngrams:
        if not ngrams_statistics.has_key(ngram):
            ngrams_statistics.update({ngram:1})
        else:
            ngram_occurrences = ngrams_statistics[ngram]
            ngrams_statistics.update({ngram:ngram_occurrences+1})
    return ngrams_statistics

def isASCII(s):
    return all(ord(char) < 128 for char in s)
