import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import pyreadr
from pyibl import Agent

# Load the RDS file
study1 = pd.read_csv("study1.csv")
study2 = pd.read_csv("study2.csv")
"""
Index(['Unnamed: 0', 'X', 'id', 'finished', 'choices', 'payoffs', 'sumPayoffs',
       'earnings', 'highPayoff', 'blockSize', 'treatment', 'labelA',
       'playedOptionA', 'playedOptionB', 'gotHighV1', 'gotLowV1', 'gotHighV2',
       'gotLowV2', 'gender', 'age', 'education', 'income', 'checkingAccount',
       'savingsAccount', 'creditCard', 'investmentAccount',
       'retirementAccount', 'noAccount', 'politics', 'politicsOther', 'ccPaid',
       'paysched', 'payschedOther', 'savingsFreq', 'retirementFreq',
       'lotteryFreq', 'description', 'gotHigh', 'gotLow', 'male', 'lottery',
       'ageRaw', 'sumRisky', 'propRisky', 'estimateRisky', 'estimateError',
       'fractionWon', 'estimateFractionWon', 'trial', 'choice', 'payoff',
       'riskyOption', 'lag1.riskyOption', 'lag2.riskyOption',
       'lag3.riskyOption', 'lag4.riskyOption', 'lag5.riskyOption',
       'lag6.riskyOption', 'lag7.riskyOption', 'lag8.riskyOption',
       'lag9.riskyOption', 'lag10.riskyOption', 'lag1.payoff', 'lag2.payoff',
       'lag3.payoff', 'lag4.payoff', 'lag5.payoff', 'lag6.payoff',
       'lag7.payoff', 'lag8.payoff', 'lag9.payoff', 'lag10.payoff',
       'switched'],
      dtype='object')
"""

#study1 = study1[study1['treatment'] == 'Clustered Feedback']
#study1 = study1[study1['treatment'] == 'Immediate Feedback' ]

columns = ["UserId", "Correct", "Treatment"]
df = pd.DataFrame([], columns=columns)

for id in study2['id'].unique():
    pdf = study2[study2['id'] == id]
    #a = Agent(name='id', default_utility=5)
    a = Agent(name='id', default_utility=0.2)
    treatment = pdf['treatment'].unique()[0]

    payoffSum = 0 
    for idx, (choice, payoff) in enumerate(zip(pdf['choice'], pdf['payoff'])):
        prediction, details = a.choose(["A", "B"], details=True)
        payoffSum += payoff 
        correct = 1 if prediction == choice else 0 

        if(treatment == 'Clustered Feedback'):
            #a.respond(correct)
            a._pending_decision = tuple((a._pending_decision[1].index(choice), a._pending_decision[1], a._pending_decision[2], a._pending_decision[3]))
            if(idx % 10 == 0):
                a.respond(payoffSum)
                payoffSum = 0
            else:
                a.respond()
        else:
            #a.respond(outcome=payoff, choice=choice)
            a.respond(correct)

        d = pd.DataFrame([[id, correct, treatment]], columns=columns)
        df = pd.concat([d, df], ignore_index=True)

df = df.groupby(['UserId', 'Treatment'], as_index=False)['Correct'].mean()
#print(df)

df = df[df['Correct'] > 0.5]

order = ['Immediate Feedback', 'Clustered Feedback']
#sns.histplot(data=df, x="Correct", hue="Treatment")
sns.violinplot(data=df, x="Treatment", order=order, y="Correct")
plt.title("IBL Model Tracing Predictive Accuracy")
plt.show()