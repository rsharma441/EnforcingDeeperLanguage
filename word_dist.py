import pandas as pd
from nltk import word_tokenize, FreqDist
from nltk.corpus import stopwords
import pos_helpers
import itertools
from nltk.corpus import names 
from collections import Counter


def string_filters(lst):

	males = [x.encode('UTF8') for x in names.words('male.txt')] 
	females = [x.encode('UTF8') for x in names.words('female.txt')] 
	males = map(lambda x: x.lower(), males)
	females = map(lambda x: x.lower(), females)

	lst = [word for word in lst if len(word) > 1]
	lst = [word for word in lst if word not in stopwords]
	lst = [word for word in lst if word not in females]
	lst = [word for word in lst if word not in males]
	return(list(filter(None, lst)))


def string_concatenate(df):
	#lst = list(itertools.product(df[0],df[1]))
	lst = [i for i in itertools.product(df[0],df[1]) if len(set(i)) == 2]
	new_list = []
	for words in lst:
		new_list.append(''.join(words))
	return(new_list)


def format_text(df, fx=None):
	if(fx!=None):
		return(df.str.lower().apply(pos_helpers.remove_punc).str.split().apply(string_filters).apply(fx))
	else:
		return(df.str.lower().apply(pos_helpers.remove_punc).str.split().apply(string_filters))

def word_dist(lst):
	cnt = Counter()
	for word in lst:
		cnt[word] += 1 
	return(pd.DataFrame(cnt.items(), columns=['word', 'count']).sort_values(by=['count', 'word'], ascending=False))

def load_dist():
	stopwords = set(stopwords.words('english'))

	data=pd.read_csv('full_data.csv', delimiter='\t')
	data.index = data['InputStoryid']
	print("Imported data from SCT Test")

	males = [x.encode('UTF8') for x in names.words('male.txt')] 
	females = [x.encode('UTF8') for x in names.words('female.txt')] 
	males = map(lambda x: x.lower(), males)
	females = map(lambda x: x.lower(), females)


	right_answers=data['rightanswer']
	wrong_answers=data['wronganswer']

	right_unigrams = format_text(right_answers).agg(sum)
	right_uni_dist = word_dist(right_unigrams)
	print(right_uni_dist)
	right_uni_dist.to_csv('word_dist/right_uni_dist.tsv' ,sep='\t')

	wrong_unigrams = format_text(wrong_answers).agg(sum)
	wrong_uni_dist = word_dist(wrong_unigrams)
	print(wrong_uni_dist)
	wrong_uni_dist.to_csv('word_dist/wrong_uni_dist.tsv' ,sep='\t')

	right_bigrams = format_text(right_answers, fx=pos_helpers.get_bigrams).agg(sum)
	right_bi_dist = word_dist(right_bigrams)
	print(right_bi_dist)
	right_bi_dist.to_csv('word_dist/right_bi_dist.tsv' ,sep='\t')

	wrong_bigrams = format_text(wrong_answers, fx=pos_helpers.get_bigrams).agg(sum)
	wrong_bi_dist = word_dist(wrong_bigrams)
	print(wrong_bi_dist)
	wrong_bi_dist.to_csv('word_dist/wrong_bi_dist.tsv' ,sep='\t')

	right_trigrams = format_text(right_answers, fx=pos_helpers.get_trigrams).agg(sum)
	right_tri_dist = word_dist(right_trigrams)
	print(right_tri_dist)
	right_tri_dist.to_csv('word_dist/right_tri_dist.tsv' ,sep='\t')

	wrong_trigrams = format_text(wrong_answers, fx=pos_helpers.get_trigrams).agg(sum)
	wrong_tri_dist = word_dist(wrong_trigrams)
	print(wrong_bi_dist)
	wrong_tri_dist.to_csv('word_dist/wrong_tri_dist.tsv' ,sep='\t')

	right_answers = data['rightanswer'].str.lower().apply(pos_helpers.remove_punc).str.split().apply(string_filters)
	right_answers.columns = ['rightanswer']
	wrong_answers = data['wronganswer'].str.lower().apply(pos_helpers.remove_punc).str.split().apply(string_filters)
	right_answers.columns = ['wronganswer']

	right_answers = right_answers.to_frame().apply(lambda x: pd.Series(x[0]),axis=1).stack().reset_index(level=1, drop=True).to_frame()
	right_answers.columns=['rightanswer']
	wrong_answers = wrong_answers.to_frame().apply(lambda x: pd.Series(x[0]),axis=1).stack().reset_index(level=1, drop=True).to_frame()
	wrong_answers.columns=['wronganswer']

	answers=right_answers.merge(wrong_answers, left_index=True, right_index=True)
	answers=answers[answers['rightanswer']!=answers['wronganswer']]
	print(answers.head())
	answers['rightwrong']=answers['rightanswer']+answers['wronganswer']
	answers_dist = answers['rightwrong'].tolist()
	answers_dist = word_dist(answers_dist).to_csv('word_dist/answers_dist.tsv' ,sep='\t')
	print(answers[answers['rightanswer']=='laid'])


right_unigrams = pd.read_csv('word_dist/right_uni_dist.tsv', delimiter='\t', index_col=0)['count'].to_dict()
wrong_unigrams = pd.read_csv('word_dist/wrong_uni_dist.tsv', delimiter='\t', index_col=0)['count'].to_dict()
right_bigrams = pd.read_csv('word_dist/right_bi_dist.tsv', delimiter='\t', index_col=0)['count'].to_dict()
wrong_bigrams = pd.read_csv('word_dist/wrong_bi_dist.tsv', delimiter='\t', index_col=0)['count'].to_dict()
right_trigrams = pd.read_csv('word_dist/right_tri_dist.tsv', delimiter='\t', index_col=0)['count'].to_dict()
wrong_trigrams = pd.read_csv('word_dist/wrong_tri_dist.tsv', delimiter='\t', index_col=0)['count'].to_dict()
answers_dist = pd.read_csv('word_dist/answers_dist.tsv', delimiter='\t', index_col=0)['count'].to_dict()
rightwrong = pd.read_csv('word_dist/rightwrong.tsv', delimiter='\t', index_col=0)


rightwrong= rightwrong[rightwrong['rightanswer']!=rightwrong['wronganswer']]
rightwrong['rightwrong']= rightwrong['rightanswer']+rightwrong['wronganswer']
rightwrong['rw_ct']= rightwrong['rightanswer']+rightwrong['wronganswer']
rightwrong=rightwrong.replace({'rightanswer': right_unigrams})
rightwrong=rightwrong.replace({'wronganswer': wrong_unigrams})
rightwrong= rightwrong[(rightwrong['rightanswer']!=1)&(rightwrong['wronganswer']!=1)]

rightwrong=rightwrong.replace({'rw_ct': answers_dist})
rightwrong['right_pct']=rightwrong['rw_ct'].apply(pd.to_numeric) / rightwrong['rightanswer'].apply(pd.to_numeric)
rightwrong['wrong_pct']=rightwrong['rw_ct'].apply(pd.to_numeric) / rightwrong['wronganswer'].apply(pd.to_numeric)
rightwrong['rw_ratio'] = 2*rightwrong['right_pct']*rightwrong['wrong_pct'] / (rightwrong['right_pct']+rightwrong['wrong_pct'] )
print(rightwrong.sort_values(by='rw_ratio', ascending=False))


