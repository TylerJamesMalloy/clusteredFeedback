import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from pyibl import Agent

# Load the RDS file
study1 = pd.read_csv("HumanBehavior/study1.csv")
study2 = pd.read_csv("HumanBehavior/study2.csv")
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

study = study2
for id in study['id'].unique():
    pdf = study[study['id'] == id]
    treatment = pdf['treatment'].unique()[0]

    a = Agent(name='id', default_utility=0.5)
    if(treatment == 'Clustered Feedback'):
        a1 = Agent(name='id', default_utility=0, decay=0, noise=0, temperature=10)
        a2 = Agent(name='id', default_utility=10, decay=1, noise=1, temperature=0.1)
        agents = [a1, a2]
        

    payoffSum = 0 
    for idx, (choice, payoff) in enumerate(zip(pdf['choice'], pdf['payoff'])):
        if(treatment == 'Clustered Feedback'):
            prediction, details = a.choose([1, 2], details=True)
            if(prediction == 1):
                prediction, details = a1.choose(["A", "B"], details=True) 
                chosen = a1 
            else:
                prediction, details = a2.choose(["A", "B"], details=True) 
                chosen = a2
        else:
            prediction, details = a.choose(["A", "B"], details=True)

        payoffSum += payoff 
        correct = 1 if prediction == choice else 0 

        if(treatment == 'Clustered Feedback'):
            for agent in agents:
                agent._pending_decision = tuple((chosen._pending_decision[1].index(choice), chosen._pending_decision[1], chosen._pending_decision[2], chosen._pending_decision[3]))
            if(idx % 10 == 0):
                for agent in agents:
                    agent.respond(payoffSum)
                payoffSum = 0
            else:
                for agent in agents:
                    agent.respond()
            a.respond(correct)
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
plt.title("HIBL Model Tracing Predictive Accuracy")
plt.show()