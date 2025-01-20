import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit

# Load the data
df = pd.read_pickle("Results/Simulations.pkl")

# Filter and group data
humans = df[df['Name'] == "Human"]
group = humans.groupby(['Agent ID'], as_index=False)['Reward'].sum()
data = group['Reward'].astype(int).to_numpy()


df['Environment'] = df['Environment'].astype(str)
df['Description'] = df['Description'].astype(str)
df['Name'] = df['Name'].astype(str)
df['Agent ID'] = df['Agent ID'].astype(str)
df['Timestep'] = df['Timestep'].astype(int)
df['Risky'] = df['Risky'].astype(int)

df['Round'] = (df['Timestep'] / 10).astype(int) 
df['Lucky'] = False
df.loc[df['Reward'] == 10, 'Lucky'] = True
# How do humans respond to a lucky round vs an unlucky round? 
# X axis, number of lucky rewards on Round N-1
# Y axis, number of risky selection on Round N
columns = ['Round N-1 Luck', 'Round N Risk', 'Description', "Environment", "Name"]
ldf = pd.DataFrame([], columns=columns)
for agent in df['Agent ID'].unique():
    agentDf = df[df['Agent ID'] == agent]
    for name in agentDf['Name'].unique():
        lastRisk = None 
        for environment in agentDf['Environment'].unique():
            envDf = agentDf[agentDf['Environment'] == environment]
            for description in envDf['Description'].unique():
                descDF = envDf[envDf['Description'] == description]
                for round in range(10):
                    roundDF = descDF[descDF['Round'] == round]
                    divisor = 1 #2 if environment == 'Immediate' else 1
                    roundLuck = np.sum(roundDF['Lucky'].to_numpy()) / divisor
                    if(round > 1):
                        d = pd.DataFrame([[roundLuck, lastRisk, description, environment, name]], columns=columns)
                        ldf = pd.concat([ldf, d], ignore_index=True)
                    lastRisk = np.sum(roundDF['Risky'].to_numpy()) / divisor

ldf['Round N-1 Luck'] = ldf['Round N-1 Luck'].astype(float)
ldf['Round N Risk'] = ldf['Round N Risk'].astype(float)