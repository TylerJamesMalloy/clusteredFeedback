
import os 
import glob 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
import numpy as np
  

path = "./Results/Participants/"
pkl_files = glob.glob(os.path.join(path, "*.pkl")) 
  
df = None
# loop over the list of csv files 
for f in pkl_files: 
    if(df is None):
        df = pd.read_pickle(f) 
    else:
        df = pd.concat([df, pd.read_pickle(f)], ignore_index=True)

df = df[(df["Environment"] != "Clustered") | (df["Timestep"] <= 29) | (df["Timestep"] >= 40)]
df = df[(df["Environment"] != "Clustered") | (df["Timestep"] <= 49) | (df["Timestep"] >= 60)]

fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(15, 6))
"""
Index(['Environment', 'Description', 'Name', 'Agent ID', 'Timestep',
       'True Reward', 'Observed Reward', 'Correct Prediction', 'Model Risky',
       'Human Risky', 'Human Reward'],
      dtype='object')
"""
for Row, Description in enumerate(["Description", "No Description"]):
    for Column, Environment in enumerate(["Immediate", "Clustered", "Delayed"]):
        pdf = df[(df['Environment'] == Environment) & (df['Description'] == Description)]
        print(Row, " ", Column)
        sns.lineplot(pdf, x='Timestep', y='Correct Prediction', hue='Name', ax=axes[Row][Column])
        #sns.lineplot(pdf, x='Timestep', y='Human Risky', label="Human", color="Green", ax=axes[Row][Column])
        axes[Row][Column].set_ylabel("")
        axes[Row][Column].set_xlabel("")

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

