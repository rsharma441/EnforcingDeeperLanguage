import pandas as pd
from scipy import stats


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




test=pd.read_csv('test.csv')
text = pd.read_csv('text.tsv', sep='\t')
full_mat = text.join(read_full_mat(test))

full_mat.to_csv('full_data.csv', sep='\t', header=True)
'''
##GLOBAL VARAIBLES
top2_indices = ['cogcomp', 'msap']
other_indices = 	full_mat_names = [ 'jonbean' , 'lizhongyang' , 'mflor', 'niko' , 'rocnlp' , 'roemmele' , 'sjtuadapt' , 'tbmihaylov' , 'teampg' , 'teamukp']
top_threshold = 0.4



test=pd.read_csv('test.csv')
text = pd.read_csv('text.tsv', sep='\t')
text.index=text['InputStoryid']
text=text.drop('InputStoryid', axis=1)
text['rightanswer_len']=text['rightanswer'].str.len()
text['wronganswer_len']=text['wronganswer'].str.len()
text['rightanswer_numwords']=text['rightanswer'].str.split().apply(len)
text['wronganswer_numwords']=text['wronganswer'].str.split().apply(len)

print("Hypothesis Test for Sentence Length", stats.ttest_ind(text['rightanswer_len'],text['wronganswer_len'],equal_var=False))
print("Hypothesis Test for Num", stats.ttest_ind(text['rightanswer_numwords'],text['wronganswer_numwords'],equal_var=False))



###TESTING TOP TWO MODELS
text['diff_len'] = text['rightanswer_len']-text['wronganswer_len']
text['diff_len_is_pos'] =  (text['rightanswer_len']-text['wronganswer_len']>0)*1
text['diff_numwords'] =  (text['rightanswer_numwords']-text['wronganswer_numwords'])
text['diff_numwords_is_pos'] =  (text['rightanswer_numwords']-text['wronganswer_numwords']>0)*1
full_mat = text.join(read_full_mat(test))
full_mat['t2r']=full_mat[top2_indices].all(axis=1).astype(int)
full_mat['one_of_t2r']=full_mat[top2_indices].any(axis=1).astype(int)
full_mat['percent_others_right'] =full_mat[other_indices].sum(axis=1)/10
t2r_tabu = full_mat[['t2r','one_of_t2r']].groupby([full_mat['percent_others_right']]).sum()
t2r_list = full_mat.loc[(full_mat[top2_indices].all(axis=1)==1)&(full_mat['percent_others_right']<top_threshold )]
other_list = full_mat.loc[(full_mat[top2_indices].all(axis=1)==1)&(full_mat['percent_others_right']>=top_threshold )]
print(full_mat.loc[full_mat['t2r']==1,[ 'diff_len' ,'diff_len_is_pos', 'diff_numwords', 'diff_numwords_is_pos' ]].groupby([full_mat['percent_others_right']]).mean())
print(full_mat.loc[full_mat['t2r']==1,[ 'diff_len' ,'diff_len_is_pos', 'diff_numwords', 'diff_numwords_is_pos' ]].groupby([full_mat['percent_others_right']]).std())

t2r_sl = t2r_list['diff_len']
other_sl = other_list['diff_len']
print(stats.ttest_ind(t2r_sl,other_sl,equal_var=False))

t2r_nw = t2r_list['diff_numwords']
other_nw = other_list['diff_numwords']
print(stats.ttest_ind(t2r_nw,other_nw,equal_var=False))

#print(full_mat.loc[full_mat['t2r']==1,[ 'diff_len' ,'diff_len_is_pos', 'diff_numwords', 'diff_numwords_is_pos' ]].groupby([full_mat['percent_others_right']]).mean())
#print(full_mat.loc[full_mat['t2r']==1,[ 'diff_len' ,'diff_len_is_pos', 'diff_numwords', 'diff_numwords_is_pos' ]].groupby([full_mat['percent_others_right']]).std())
#print(full_mat.loc[(full_mat[top2_indices].all(axis=1)==1)&(full_mat['percent_others_right']<0.3 )])
'''
