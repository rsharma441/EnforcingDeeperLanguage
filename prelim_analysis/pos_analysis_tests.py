import pandas as pd
from scipy import stats
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from statsmodels.stats.proportion import proportions_ztest
import math

def read_full_mat(test):

	cogcomp=pd.read_csv('../model_outputs/cogcomp_answer.txt' )
	jonbean=pd.read_csv('../model_outputs/jonbean_answer.txt')
	lizhongyang=pd.read_csv('../model_outputs/lizhongyang_answer.txt')
	mflor=pd.read_csv('../model_outputs/mflor_answer.txt')
	msap=pd.read_csv('../model_outputs/msap_answer.txt')
	niko=pd.read_csv('../model_outputs/niko_answer.txt')
	rocnlp=pd.read_csv('../model_outputs/rocnlp_answer.txt')
	roemmele=pd.read_csv('../model_outputs/roemmele_answer.txt')
	sjtuadapt=pd.read_csv('../model_outputs/sjtuadapt_answer.txt')
	tbmihaylov=pd.read_csv('../model_outputs/tbmihaylov_answer.txt')
	teampg=pd.read_csv('../model_outputs/teampg_answer.txt')
	teamukp=pd.read_csv('../model_outputs/teamukp_answer.txt')

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


##GLOBAL VARAIBLES
top2_indices = ['cogcomp', 'msap']
other_indices = 	full_mat_names = [ 'jonbean' , 'lizhongyang' , 'mflor', 'niko' , 'rocnlp' , 'roemmele' , 'sjtuadapt' , 'tbmihaylov' , 'teampg' , 'teamukp']
poslist = ['PRP$','VBG','VBD','VBN',',',"''",'VBP','WDT','JJ','WP','VBZ','DT','RP','$','NN','POS','.','TO','PRP','RB','NNS','NNP','``','WRB','CC','PDT','RBR','CD','EX','IN','MD','NNPS','JJS','JJR','VB','UH','RBS',':','FW']

top_threshold = 0.4



test=pd.read_csv('test.csv')
text = pd.read_csv('text.tsv', sep='\t')
text.index=text['InputStoryid']
text=text.drop('InputStoryid', axis=1)

full_mat = text.join(read_full_mat(test))

full_mat['t2r']=full_mat[top2_indices].all(axis=1).astype(int)
full_mat['one_of_t2r']=full_mat[top2_indices].any(axis=1).astype(int)
full_mat['percent_others_right'] =full_mat[other_indices].sum(axis=1)/10


full_mat['pos_seq']=full_mat.apply(lambda row:   " ".join([x[1] for x in pos_tag(word_tokenize(row['rightanswer']))]), axis=1)


right_answers_pos=full_mat.apply(lambda row:   " ".join([x[1] for x in pos_tag(word_tokenize(row['rightanswer']))]), axis=1).to_frame()
print(right_answers_pos)
right_answers_pos.columns =['pos_seq']
vectorizer_r = CountVectorizer()
Xr = vectorizer_r.fit_transform(right_answers_pos['pos_seq'])
pos_headers_r = vectorizer_r.get_feature_names()
right_answers_pos = right_answers_pos.join(pd.DataFrame(Xr.toarray(), index =text.index, columns= pos_headers_r))


wrong_answers_pos=full_mat.apply(lambda row:   " ".join([x[1] for x in pos_tag(word_tokenize(row['wronganswer']))]), axis=1).to_frame()
wrong_answers_pos.columns =['pos_seq']
vectorizer_w = CountVectorizer()
Xw = vectorizer_w.fit_transform(wrong_answers_pos['pos_seq'])
pos_headers_w = vectorizer_w.get_feature_names()
wrong_answers_pos = wrong_answers_pos.join(pd.DataFrame(Xw.toarray(), index =text.index, columns= pos_headers_w))


pos_headers_total = pos_headers_r + [i for i in pos_headers_w if i not in pos_headers_r]

n1 = right_answers_pos.shape[0]
n2 = wrong_answers_pos.shape[0]

for pos in pos_headers_total:
	if pos not in pos_headers_r:
		print(pos, "is only in Wrong Answers") 
	elif pos not in pos_headers_w:
		print(pos, "is only in Right Answers")
	else: 
		print(pos)
		try:
			x1 = right_answers_pos[pos].sum(axis=0)
			x2 = wrong_answers_pos[pos].sum(axis=0)
			p_star = float(x1+x2)/float(n1+n2)
			p1 = right_answers_pos[pos].mean(axis=0)
			p2 = wrong_answers_pos[pos].mean(axis=0)
			#print(x1, x2, p_star, p1,p2)
			z = (p1-p2)/(math.sqrt((p_star*(1-p_star)*((1.0/float(n1)+(1.0/float(n2)))))))
			print(z)

		except ValueError:
			print("INVALID ENTRIES")

''''


t2r_list = full_mat.loc[(full_mat[top2_indices].all(axis=1)==1)&(full_mat['percent_others_right']<top_threshold )]
other_list = full_mat.loc[(full_mat[top2_indices].all(axis=1)==1)&(full_mat['percent_others_right']>=top_threshold )]

t2r_pos_means = t2r_list[vectorizer.get_feature_names()].mean(axis=0)
t2r_pos_std= t2r_list[vectorizer.get_feature_names()].std(axis=0)

other_pos_means = other_list[vectorizer.get_feature_names()].mean(axis=0)
other_pos_std= other_list[vectorizer.get_feature_names()].std(axis=0)

#print((t2r_pos_means+t2r_pos_std)-other_pos_means)

t2size = t2r_list.shape[0]
othersize = other_list.shape[0]
for pos in pos_headers:
	print(pos)
	try:
		p = ((t2r_pos_means[pos] * t2size) + (other_pos_means[pos] * othersize))/(t2size+othersize)
		se = math.sqrt((p*(1-p)*((1.0/float(t2size)+(1.0/float(othersize))))))
		z = (t2r_pos_means[pos]-other_pos_means[pos])/se
		print("Z value is ", z )
	except ValueError:
	    print "INVALID ENTRIES"

'''




