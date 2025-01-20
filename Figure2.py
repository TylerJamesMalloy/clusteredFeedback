import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit

ldf = pd.read_pickle("./Results/RiskReward.pkl")
fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(15, 6))
orders = [ [[2,2,1], [2,2,1], [2,2,1]],  [[1,1,1], [1,1,1], [1,1,1]]]
for row, desc in enumerate(["Description", "No Description"]):
    for col, envr in enumerate(["Immediate", "Clustered", "Delayed"]):
        data = ldf[(ldf['Description'] == desc) & (ldf['Environment'] == envr)] 
        human = data[data["Name"] == "Human"].groupby(['Round N-1 Luck'], as_index=False)["Round N Risk"].mean()
        hibl = data[data["Name"] == "HIBL"].groupby(['Round N-1 Luck'], as_index=False)["Round N Risk"].mean()
        ibl = data[data["Name"] == "IBL"].groupby(['Round N-1 Luck'], as_index=False)["Round N Risk"].mean()
        sns.regplot(human, x="Round N-1 Luck", y="Round N Risk", label="Human", order=orders[row][col][0], ax=axes[row][col])
        sns.regplot(hibl, x="Round N-1 Luck", y="Round N Risk", label="HIBL", order=orders[row][col][1], ax=axes[row][col])
        sns.regplot(ibl, x="Round N-1 Luck", y="Round N Risk", label="IBL", order=orders[row][col][2], ax=axes[row][col])
        axes[row][col].set_ylabel("")
        axes[row][col].set_xlabel("")
        axes[row][col].set_xlim(-1,11)
        axes[row][col].set_ylim(-1,11)

axes[0][0].set_title("Immediate Feedback", fontsize=20)
axes[0][1].set_title("Clustered Feedback", fontsize=20)
axes[0][2].set_title("Delayed Feedback", fontsize=20)

axes[0][0].set_ylabel("Round N Risky Choices", fontsize=14)
axes[1][0].set_ylabel("Round N Risky Choices", fontsize=14)

axes[1][0].set_xlabel("Round N-1 Lucky Rewards", fontsize=14)
axes[1][1].set_xlabel("Round N-1 Lucky Rewards", fontsize=14)
axes[1][2].set_xlabel("Round N-1 Lucky Rewards", fontsize=14)

axes[0][0].legend()

fig.text(0.02, 0.28, 'No Description', va='center', rotation='vertical', fontsize=20)
fig.text(0.02, 0.7, 'Description', va='center', rotation='vertical', fontsize=20)

plt.subplots_adjust(left=0.08, bottom=0.085, right=0.995, top=0.875, wspace=0, hspace=0)
plt.suptitle("Round N Risky Choices by Round N-1 Lucky Rewards", fontsize=22)

plt.show()

"""
# Load the data
df = pd.read_pickle("Results/Simulations.pkl")
human = df[df["Name"] == "Human"]
df1 = pd.read_pickle("Results/HIBL.pkl")
df2 = pd.read_pickle("Results/IBL.pkl")

df['Lucky'] = 0
human.loc[(human['Reward'] == 10), 'Lucky'] = 1
group = df.groupby(['Agent ID', 'Description', "Environment", "Name", 'Lucky'], as_index=False)['Risky'].mean()
high = group[group['Risky'] >= 0.9]
low = group[group['Risky'] <= 0.1]
df = group[(group['Risky'] < 0.9) & (group['Risky'] > 0.1)]
#df = df[df['Agent ID'].isin(df['Agent ID'].unique())]

df = pd.concat([human, df1, df2])

df['Environment'] = df['Environment'].astype(str)
df['Description'] = df['Description'].astype(str)
df['Name'] = df['Name'].astype(str)
df['Agent ID'] = df['Agent ID'].astype(str)
df['Timestep'] = df['Timestep'].astype(int)
df['Risky'] = df['Risky'].astype(int)
df['Lucky'] = 0
df.loc[df['Reward'] == 10, 'Lucky'] = 1
# How do humans respond to a lucky round vs an unlucky round? 
# X axis, number of lucky rewards on Round N-1
# Y axis, number of risky selection on Round N
columns = ['Round N-1 Luck', 'Round N Risk', 'Description', "Environment", "Name"]
ldf = pd.DataFrame([], columns=columns)
for agent in df['Agent ID'].unique():
    agentDf = df[df['Agent ID'] == agent]
    for name in agentDf['Name'].unique():
        nameDF = agentDf[agentDf['Name'] == name]
        for environment in nameDF['Environment'].unique():
            envDf = nameDF[nameDF['Environment'] == environment]
            for description in envDf['Description'].unique():
                lastRisk = None 
                descDF = envDf[envDf['Description'] == description]
                lastTs = 0
                for nextTs in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                    roundDF = descDF[(descDF['Timestep'] > lastTs) & (descDF['Timestep'] <= nextTs)]
                    roundLuck = np.sum(roundDF['Lucky'].to_numpy()[0:10]) 
                    if(lastTs >= 10):
                        d = pd.DataFrame([[roundLuck, lastRisk, description, environment, name]], columns=columns)
                        ldf = pd.concat([ldf, d], ignore_index=True)
                    lastRisk = np.sum(roundDF['Risky'].to_numpy()[0:10])
                    lastTs = nextTs


ldf['Round N-1 Luck'] = ldf['Round N-1 Luck'].astype(float)
ldf['Round N Risk'] = ldf['Round N Risk'].astype(float)
ldf.to_pickle("./Results/RiskReward.pkl")
"""