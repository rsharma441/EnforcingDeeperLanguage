import pandas as pd

from nltk import pos_tag
from nltk.tokenize import word_tokenize
import pandas as pd

import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from sklearn.model_selection import train_test_split


from sentiment_analysis import convert_to_sequence


def read_full_mat(test):

	cogcomp=pd.read_csv('cogcomp_answer.txt' )
	jonbean=pd.read_csv('jonbean_answer.txt')
	lizhongyang=pd.read_csv('lizhongyang_answer.txt')
	mflor=pd.read_csv('mflor_answer.txt')
	msap=pd.read_csv('msap_answer.txt')
	niko=pd.read_csv('niko_answer.txt')
	rocnlp=pd.read_csv('rocnlp_answer.txt')
	roemmele=pd.read_csv('roemmele_answer.txt')
	sjtuadapt=pd.read_csv('sjtuadapt_answer.txt')
	tbmihaylov=pd.read_csv('tbmihaylov_answer.txt')
	teampg=pd.read_csv('teampg_answer.txt')
	teamukp=pd.read_csv('teamukp_answer.txt')

	full_mat = pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(test, cogcomp, left_on = 'InputStoryid', right_on = 'InputStoryid'), \
		jonbean, left_on = 'InputStoryid', right_on = 'InputStoryid'), \
		lizhongyang, left_on = 'InputStoryid', right_on = 'InputStoryid'), \
		mflor, left_on = 'InputStoryid', right_on = 'InputStoryid'), \
		msap, left_on = 'InputStoryid', right_on = 'InputStoryid'), \
		niko, left_on = 'InputStoryid', right_on = 'InputStoryid'), \
		rocnlp, left_on = 'InputStoryid', right_on = 'InputStoryid'), \
		roemmele, left_on = 'InputStoryid', right_on = 'InputStoryid'), \
		sjtuadapt, left_on = 'InputStoryid', right_on = 'InputStoryid'), \
		tbmihaylov, left_on = 'InputStoryid', right_on = 'InputStoryid'), \
		teampg, left_on = 'InputStoryid', right_on = 'InputStoryid'), \
		teamukp, left_on = 'InputStoryid', right_on = 'InputStoryid')

	full_mat.index=full_mat['InputStoryid']
	full_mat_names = ['storyid', 'answer', 'cogcomp' , 'jonbean' , 'lizhongyang' , 'mflor' , 'msap' , 'niko' , 'rocnlp' , 'roemmele' , 'sjtuadapt' , 'tbmihaylov' , 'teampg' , 'teamukp']
	full_mat.columns = full_mat_names
	for name in full_mat_names[2:]:
		full_mat[name]= (full_mat[name] == full_mat['answer']).astype(int)
	full_mat=full_mat.drop('storyid', axis=1)
	full_mat=full_mat.drop('answer', axis=1)

	return(full_mat)


def return_top2right(full_mat):
	top2right = full_mat[full_mat['cogcomp']*full_mat['msap']==1]
	top2rightpcts = (top2right.drop(['cogcomp', 'msap'], axis=1)).mean(1).sort_values()
	t2rp_full = top2rightpcts[top2rightpcts <= 1].to_frame().join(text, how="inner").join(full_mat, how='inner')
	return(t2rp_full)

def return_top2wrong(full_mat):
	top2wrong = full_mat[full_mat['cogcomp']+full_mat['msap']==0]
	top2wrongpcts = (top2wrong.drop(['cogcomp', 'msap'], axis=1)).mean(1).sort_values()
	t2wp_full = top2wrongpcts[top2wrongpcts >= 0].to_frame().join(text, how="inner").join(full_mat, how='inner')
	return(t2wp_full)


def add_wordlength(data):
	data['rightanswer_len']=data['rightanswer'].str.len()
	data['wronganswer_len']=data['wronganswer'].str.len()
	data['diff_len'] =  (data['rightanswer_len']-data['wronganswer_len']>0)*1
	return(data)


def add_wordct(data):
	data['rightanswer_numwords']=data['rightanswer'].str.split().apply(len)
	data['wronganswer_numwords']=data['wronganswer'].str.split().apply(len)
	data['diff_numwords'] =  (data['rightanswer_numwords']-data['wronganswer_numwords']>0)*1
	return(data)




test=pd.read_csv('test.csv')
sentiments=pd.read_csv('sentiment_scores.tsv', sep='\t')
sentiments.index=sentiments['storyid']

text = pd.read_csv('text.tsv', sep='\t')
text.index=text['InputStoryid']
text=text.drop('InputStoryid', axis=1)

full_mat = read_full_mat(test)

###questions with only top 2 got right
t2rp_full = return_top2right(full_mat)

###questions with only top 2 got wrong
t2wp_full = return_top2wrong(full_mat)


###all right/all wrong
allr_full = text.join(full_mat[full_mat.all(axis=1)], how="inner")
allw_full= text.join(full_mat[(1-full_mat).all(axis=1)], how="inner")


###sentence length of
t2rp_full = add_wordlength(t2rp_full)
allr_full = add_wordlength(allr_full)

###sentence numbers of words

t2rp_full = add_wordct(t2rp_full)
allr_full = add_wordct(allr_full)


##CSV OUTPUT
t2rp_full.to_csv("t2rp_full_output.csv", sep='\t', header=True, index=True)
allr_full.to_csv("allr_full_output.csv", sep='\t', header=True, index=True)
t2wp_full.to_csv("t2wp_full_output.csv", sep='\t', header=True, index=True)
allw_full.to_csv("allw_full_output.csv", sep='\t', header=True, index=True)






print(t2rp_full.groupby([t2rp_full[0]]).mean())
t2rp_full.groupby([t2rp_full[0]]).mean().to_csv("string_lengths.csv", sep='\t', header=True, index=True)
t2rp_full.groupby([t2rp_full[0]]).std().to_csv("string_lengths_std.csv", sep='\t', header=True, index=True)
t2rp_full.groupby([t2rp_full[0]]).size().to_csv("string_lengths_groupsizes.csv", sep='\t', header=True, index=True)



sentiments=sentiments.join(t2rp_full[0], how="inner")
print(sentiments.groupby([sentiments[0]]).mean())
sentiments.groupby([sentiments[0]]).mean().to_csv("sentiment_analysis.csv", sep='\t', header=True, index=True)



seed = 7
numpy.random.seed(seed)

pos = text
pos['target'] = 0
pos.loc[t2rp_full.index.values.tolist(), 'target']=1
max_length = 17

rightanswer_list = convert_to_sequence(pos['rightanswer'])
print(rightanswer_list)
rightanswer_list = sequence.pad_sequences(rightanswer_list, maxlen=max_length)

wronganswer_list = convert_to_sequence(pos['wronganswer'])
wronganswer_list = sequence.pad_sequences(wronganswer_list, maxlen=max_length)


X_train, X_test, y_train, y_test = train_test_split(rightanswer_list, pos['target'], test_size=0.33, random_state=seed)


input_length = X_train.shape[0]
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(input_length, embedding_vecor_length, input_length= max_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)


# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


