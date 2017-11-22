from nltk import pos_tag
from nltk.tokenize import word_tokenize
import pandas as pd

# LSTM with Dropout for sequence classification in the IMDB dataset
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence



def convert_to_sequence(list):
	map={'PRP$': 1, 'VBG': 2, 'VBD': 3, 'VBN': 7, ',': 25, "''": 8, 'VBP': 9, 'WDT': 10, 'JJ': 12, 'WP': 13, 'VBZ': 14, 'DT': 15, 'RP': 19, '$': 20, 'NN': 21, 'POS': 22, '.': 26, 'TO': 28, 'PRP': 30, 'RB': 31, 'NNS': 33, 'NNP': 35, '``': 6, 'WRB': 0, 'CC': 4, 'PDT': 5, 'RBR': 23, 'CD': 16, 'EX': 11, 'IN': 32, 'MD': 17, 'NNPS': 18, 'JJS': 24, 'JJR': 34, 'VB': 27, 'UH': 29, 'RBS':36, ':':37, 'FW':38}

	tokenlist = []
	for w in list:
		tokens_converted = []
		index = 0
		tokens = word_tokenize(w) # Generate list of tokens
		tokens_pos = pos_tag(tokens) 
		for t in tokens_pos:
			tokens_converted.append(map[t[1]])

		tokenlist.append(tokens_converted)
	return(tokenlist)

