from scipy import stats
import pandas as pd
import nltk

from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff


data=pd.read_csv('full_data.csv', delimiter='\t')
print("Imported data from SCT Test")
data.index = data['InputStoryid']
right_answers=data['rightanswer']
wrong_answers=data['wronganswer']
print('Right Answers:', right_answers.head())
print('Wrong Answers:', wrong_answers.head())

##sentence legnth
right_len=right_answers.str.len()
wrong_len=wrong_answers.str.len()
print('Right Answers:', right_len.head())
print('Wrong Answers:', wrong_len.head())

##number of words
right_numwords=right_answers.apply(nltk.word_tokenize).apply(len)
wrong_numwords=wrong_answers.apply(nltk.word_tokenize).apply(len)
print('Right Answers:', right_numwords.head())
print('Wrong Answers:', wrong_numwords.head())

##output to file
right_sent = right_len.to_frame().join(right_numwords.to_frame(), rsuffix='_x')
right_sent.columns = ['length', 'num_words']
right_sent.to_csv('right_sent.csv', '\t')


##hypothesis testing
length_test = stats.ttest_ind(right_len,wrong_len,equal_var=False)
numwords_test = stats.ttest_ind(right_numwords,wrong_numwords,equal_var=False)
print('Avg Length of Right Ans:' , right_len.mean(), 'Avg Length of Wrong Ans:' , wrong_len.mean())
print('Avg NumWords of Right Ans:' , right_numwords.mean(), 'Avg NumWords of Wrong Ans:' , wrong_numwords.mean())
print('Length Hypothesis Test:', length_test)
print('Num Words Test:', numwords_test)

###plots
answers = [right_len.as_matrix(),wrong_len.as_matrix()]
labels = ['right_answer_len', 'wrong_answer_len']
fig = ff.create_distplot(answers, labels)
py.plot(fig, filename='Plot of Answers Length')
sent = [right_sent.as_matrix(),wrong_sent.as_matrix()]
sent_labels = ['right_sent', 'wrong_sent']
fig = ff.create_distplot(sent, sent_labels)
py.plot(fig, filename='Plot of Sentiment')

