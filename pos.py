import pandas as pd
import re, string
import math
from scipy import stats
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from statsmodels.stats.proportion import proportions_ztest
import pos_helpers 


poslist = ['PRP$','VBG','VBD','VBN',',',"''",'VBP','WDT','JJ','WP','VBZ','DT','RP','$','NN','POS','.','TO','PRP','RB','NNS','NNP','``','WRB','CC','PDT','RBR','CD','EX','IN','MD','NNPS','JJS','JJR','VB','UH','RBS',':','FW']


data=pd.read_csv('full_data.csv', delimiter='\t')
print("Imported data from SCT Test")
data.index = data['InputStoryid']
right_answers=data['rightanswer'].apply(pos_helpers.remove_punc)
wrong_answers=data['wronganswer'].apply(pos_helpers.remove_punc)
print('Right Answers:', right_answers.head())
print('Wrong Answers:', wrong_answers.head())
 

right_answers_pos=pd.DataFrame(right_answers.apply(word_tokenize).apply(pos_tag).apply(pos_helpers.first_element_bucket))
right_answers_pos['unigrams']=right_answers_pos['rightanswer'].apply(lambda x: ' '.join(x))
right_answers_pos['bigrams'] = right_answers_pos['rightanswer'].apply(pos_helpers.get_bigrams).apply(lambda x: ' '.join(x))
#right_answers_pos['trigrams'] = right_answers_pos['rightanswer'].apply(pos_helpers.get_trigrams).apply(lambda x: ' '.join(x))

wrong_answers_pos=pd.DataFrame(wrong_answers.apply(word_tokenize).apply(pos_tag).apply(pos_helpers.first_element_bucket))
wrong_answers_pos['unigrams']=wrong_answers_pos['wronganswer'].apply(lambda x: ' '.join(x))
wrong_answers_pos['bigrams'] = wrong_answers_pos['wronganswer'].apply(pos_helpers.get_bigrams).apply(lambda x: ' '.join(x))
#wrong_answers_pos['trigrams'] = wrong_answers_pos['wronganswer'].apply(pos_helpers.get_trigrams).apply(lambda x: ' '.join(x))

print(right_answers_pos)
print(wrong_answers_pos)
headers = {}

right_uni_pos_matrix, headers['right_uni'] = pos_helpers.create_pos_matrix(right_answers_pos['unigrams'], data.index)
wrong_uni_pos_matrix, headers['wrong_uni'] = pos_helpers.create_pos_matrix(wrong_answers_pos['unigrams'], data.index)
right_bi_pos_matrix, headers['right_bi'] = pos_helpers.create_pos_matrix(right_answers_pos['bigrams'], data.index)
wrong_bi_pos_matrix, headers['wrong_bi'] = pos_helpers.create_pos_matrix(wrong_answers_pos['bigrams'], data.index)
#right_tri_pos_matrix, headers['right_tri'] = pos_helpers.create_pos_matrix(right_answers_pos['trigrams'], data.index)
#wrong_tri_pos_matrix, headers['wrong_tri'] = pos_helpers.create_pos_matrix(wrong_answers_pos['trigrams'], data.index)


right_pos_matrix = right_uni_pos_matrix.join(right_bi_pos_matrix)
wrong_pos_matrix = wrong_uni_pos_matrix.join(wrong_bi_pos_matrix)

pos_sizes = {}
pos_sizes['r_uni'] = right_answers_pos['unigrams'].str.count(" ")+1
pos_sizes['r_bi'] = right_answers_pos['bigrams'].str.count(" ")+1
#pos_sizes['r_tri'] = right_answers_pos['trigrams'].str.count(" ")+1
pos_sizes['w_uni'] = wrong_answers_pos['unigrams'].str.count(" ")+1
pos_sizes['w_bi'] = wrong_answers_pos['bigrams'].str.count(" ")+1
#pos_sizes['w_tri'] = wrong_answers_pos['trigrams'].str.count(" ")+1

###hypothesis testing
compare_dist=pos_helpers.compare_pos_dist(headers, pos_sizes, right_pos_matrix, wrong_pos_matrix)

### output to file
compare_dist.to_csv('pos_dist_bucket.csv', '\t')



