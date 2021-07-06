# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 10:05:17 2021

@author: NicNeo
"""

import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


# run for the first time
#nltk.download('punkt')

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

''' test code
a = "How long does shipping take?"
print(a)
a = tokenize(a)
print(a)

words = ['Organize', 'organizes', 'organiing']
stemmed_words = [stem(w) for w in words]
print(stemmed_words)

sentence = ["hello", "how", "are", "you"]
words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
bag = bag_of_words(sentence, words)
print(bag)
'''

