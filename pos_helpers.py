import re, string, math
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def remove_punc(s):
	return(re.sub('[%s]' % re.escape(string.punctuation), '', s))

def first_element(lst):
	newlist = []
	for item in lst:
		if '$' not in item[1]:
			newlist.append(item[1])
	return(newlist)

def first_element_bucket(lst):

	pos_map = {'CC':'CC','CD':'CD','DT':'DT','EX':'EX','FW':'FW','IN':'IN','JJ':'JJ','JJR':'JJ','JJS':'JJ','MD':'MD','NN':'NN','NNP':'NN','NNPS':'NN','NNS':'NN','PDT':'PDT','POS':'POS','PRP':'PRP','PRP':'PRP','RB':'RB','RBR':'RB','RBS':'RB','RP':'RP','TO':'TO','UH':'UH','VB':'VB','VBD':'VB','VBG':'VB','VBN':'VB','VBP':'VB','VBZ':'VB','WDT':'WDT','WP':'WP','WRB':'WRB'}

	newlist = []
	for item in lst:
		if '$' not in item[1]:
			newlist.append(pos_map[item[1]])
	return(newlist)

def get_bigrams(lst):
	return([a + b for a,b in zip(lst, lst[1:])])

def get_trigrams(lst):
	return([a + b +c for a,b,c in zip(lst, lst[1:], lst[2:])])

def create_pos_matrix(input_mat, i):
	vec = CountVectorizer()
	X = vec.fit_transform(input_mat)
	headers = vec.get_feature_names()
	pos_matrix = pd.DataFrame(X.toarray(), index=i, columns= headers).div(input_mat.str.count(" ")+1, axis=0)
	pos_matrix.index = i
	return(pos_matrix, headers)

def compare_pos_dist(headers, pos_sizes, right_matrix, wrong_matrix ):

	headers_right = headers['right_uni'] + [i for i in headers['right_bi'] if i not in headers['right_uni']]
	headers_wrong = headers['wrong_uni'] + [i for i in headers['wrong_bi'] if i not in headers['wrong_uni']]	
	headers_total = headers_right + [i for i in headers_wrong if i not in headers_right]


	n1 = right_matrix.shape[0]
	n2 = wrong_matrix.shape[0]
	final_list=[]
	for pos in headers_total:
		pos_dict={}
		pos = pos.encode('utf-8')
		if pos not in headers_right:
			pos_dict['pos']=pos
			pos_dict['code'] = 'W'
			pos_dict['r_ct']=0
			if pos in headers['wrong_uni']:
				pos_dict['w_ct']=wrong_matrix[pos].multiply(pos_sizes['w_uni']).sum(axis=0)
			else:
				pos_dict['w_ct']=wrong_matrix[pos].multiply(pos_sizes['w_bi']).sum(axis=0)

		elif pos not in headers_wrong:
			pos_dict['pos']=pos
			pos_dict['code']='R'
			pos_dict['w_ct']=0
			if pos in headers['right_uni']:
				pos_dict['r_ct']=right_matrix[pos].multiply(pos_sizes['r_uni']).sum(axis=0)
			else:
				pos_dict['r_ct']=right_matrix[pos].multiply(pos_sizes['r_bi']).sum(axis=0)

		else: 
			x1 = right_matrix[pos].sum(axis=0)
			x2 = wrong_matrix[pos].sum(axis=0)
			p1 = right_matrix[pos].mean(axis=0)/n1
			p2 = wrong_matrix[pos].mean(axis=0)/n2
			p_star = float(x1+x2)/float(n1+n2)
			num = p1 - p2
			denom = math.sqrt((p_star)*(1-p_star)*((1.0/float(n1))+(1.0/float(n2))))
			z = num/denom
			pos_dict['pos']=pos
			pos_dict['z']=z
			pos_dict['code']='B'
			if pos in headers['wrong_uni']:
				pos_dict['w_ct']=wrong_matrix[pos].multiply(pos_sizes['w_uni']).sum(axis=0)
			else:
				pos_dict['w_ct']=wrong_matrix[pos].multiply(pos_sizes['w_bi']).sum(axis=0)
			
			if pos in headers['right_uni']:
				pos_dict['r_ct']=right_matrix[pos].multiply(pos_sizes['r_uni']).sum(axis=0)
			else:
				pos_dict['r_ct']=right_matrix[pos].multiply(pos_sizes['r_bi']).sum(axis=0)

		final_list.append(pos_dict)

	return(pd.DataFrame(final_list))
