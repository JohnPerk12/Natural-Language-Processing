#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 20:05:15 2020

@author: johnperkins
"""

import os
import pandas as pd
from glob import glob


# Analysis without file conversion to include date. .txt file cannot be transposed into standard csv
# continue with glob of analysis
# EDA remainders below.
os.chdir('/Users/pathway.../washington')

filenames = glob("*.txt")

with open("output_file.txt", "w") as outfile:
    for filename in filenames:
        with open(filename) as infile:
            contents = infile.read()
            outfile.write(contents)
            
import spacy
from spacy.lang.en import English
parser = English()

def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    
from nltk.stem.wordnet import WordNetLemmatizer
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

nltk.download('stopwords')
en_stop = nltk.corpus.stopwords.words('english')

newStopWords = ['adams', 'washington','representative','gentleman',' representative',
                'date="december"','date="november"','tobias']
en_stop.extend(newStopWords)

def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens

import random
text_data = []
with open('output_file.txt') as f:
    for line in f:
        tokens = prepare_text_for_lda(line)
        if random.random() > .99:
            print(tokens)
            text_data.append(tokens)

from gensim import corpora
dictionary = corpora.Dictionary(text_data)
corpus = [dictionary.doc2bow(text) for text in text_data]

import pickle
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')

import gensim
NUM_TOPICS = 1
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
ldamodel.save('model5.gensim')
topics = ldamodel.print_topics(num_words=5)
for topic in topics:
    print(topic)



bigram = gensim.models.Phrases('output_file.txt', min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram['outfile.txt'], threshold=100)  

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


#---------------------------------------------------------------------------------------------------------------------

#Analysis for attempting file conversion. Non standard data

read_file = pd.read_csv('/Users/pathway.../washington/washington_speeches_000.txt')
read_file.to_csv ('/Users/pathway.../washington_speeches_000.csv')

df = pd.read_fwf('washington_speeches_000.txt')
df.to_csv('washington_speeches_000.csv')



os.chdir('/Users/pathway.../washington')

#.txt files fundamentally cannot be converted to csv
# need csv (or like file) to link year for period analysis - come to later.

data = pd.read_csv("washington_speeches_000.txt", delim_whitespace=True , header = None, index_col = 0)
data = data.dropna()
data = data.transpose()
data.to_csv("output.csv", index = False)

os.chdir('/Users/pathway.../washington')


#possibly develop own corpus?
import os, os.path
path = os.path.expanduser('~/nltk_data')
import nltk.data
path in nltk.data.path

'~/nltk_data/corpora/your_corpus'


#---------------------------------------------------------------------------------------------------------------------
#remove the '<' and '>' brackets as well as line breaks to output text. pandas failed to convert txt to readable file.
#maybe a work around to pandas failing to convert?
import re
import os
import pandas as pd
from glob import glob


os.chdir('/Users/pathway.../washington')
filenames = glob("*.txt")

with open("output_file.txt", "w") as outfile2:
    for filename in filenames:
        with open(filename, encoding='utf-8') as infile:
            contents = re.sub(r"<[^>]*>", " ", contents)
            outfile2.write(contents)

output2 = open("outfile2","r")

string_without_line_breaks =""
for line in output2:
    stripped_line = line.strip()
    string_without_line_breaks += stripped_line
output2.close()

print(string_without_line_breaks)
df = pd.read_csv('output_file.txt', error_bad_lines=False)



            
