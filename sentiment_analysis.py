from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from scipy import stats


data=pd.read_csv('full_data.csv')
print("Imported data from SCT Test")
data.index = data['InputStoryid']
right_answers=data['rightanswer'].to_frame()
wrong_answers=data['wronganswer'].to_frame()
print('Right Answers:', right_answers.head())
print('Wrong Answers:', wrong_answers.head())

##sentiment scoring
analyzer = SentimentIntensityAnalyzer()
headers= ['neg', 'neu', 'pos', 'compound']

right_answers = right_answers.join(pd.DataFrame([analyzer.polarity_scores(sentence) for sentence in right_answers['rightanswer']], index=data.index, columns=headers))
wrong_answers = wrong_answers.join(pd.DataFrame([analyzer.polarity_scores(sentence) for sentence in wrong_answers['wronganswer']], index=data.index, columns=headers))

print(right_answers.head())
print(wrong_answers.head())

##output to file
right_answers.to_csv('right_sentiment.csv', sep='\t')
wrong_answers.to_csv('wrong_sentiment.csv', sep='\t')

print('Avg Neg of Right Ans:' , right_answers['neg'].mean(), 'Avg Neg of Wrong Ans:' , wrong_answers['neg'].mean())
print('Avg Neu of Right Ans:' , right_answers['neu'].mean(), 'Avg Neu of Wrong Ans:' , wrong_answers['neu'].mean())
print('Avg Pos of Right Ans:' , right_answers['pos'].mean(), 'Avg Pos of Wrong Ans:' , wrong_answers['pos'].mean())
print('Avg Comp of Right Ans:' , right_answers['compound'].mean(), 'Avg Comp of Wrong Ans:' , wrong_answers['compound'].mean())

neg_test = stats.ttest_ind(right_answers['neg'],wrong_answers['neg'],equal_var=False)
neu_test = stats.ttest_ind(right_answers['neu'],wrong_answers['neu'],equal_var=False)
pos_test = stats.ttest_ind(right_answers['pos'],wrong_answers['pos'],equal_var=False)
comp_test = stats.ttest_ind(right_answers['compound'],wrong_answers['compound'],equal_var=False)

print('Neg Hypothesis Test:', neg_test)
print('Neu Hypothesis Test:', neu_test)
print('Pos Hypothesis Test:', pos_test)
print('Comp Words Test:', comp_test)