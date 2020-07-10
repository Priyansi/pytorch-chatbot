import nltk
import numpy as np
from nltk.stem import PorterStemmer

# nltk.download('punkt')


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    stemmer = PorterStemmer()
    return stemmer.stem(word.lower())


def bag_of_words(stemmed_sentence, all_words):
    bog = np.zeros(len(all_words), dtype=np.float32)
    for ind, word in enumerate(all_words):
        if word in stemmed_sentence:
            bog[ind] = 1.0
    return bog
