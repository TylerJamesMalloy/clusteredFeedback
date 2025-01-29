
import os 
import glob 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
import numpy as np
# 69.4
# 62.5 
  
fig, axes = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(15, 5))

df = pd.read_pickle("./Results/ModelTracing.pkl")

group = df.groupby(['Environment', 'Description', 'Name', 'Agent ID'], as_index=False)["Correct Prediction"].mean()
dec = group[group['Description'] == "Description"]
noDec = group[group['Description'] == "No Description"]

sns.barplot(data=dec  , x="Environment", y="Correct Prediction", hue="Name", ax=axes[0], order=['Immediate', 
'Clustered', 'Delayed'])
sns.barplot(data=noDec, x="Environment", y="Correct Prediction", hue="Name", ax=axes[1], order=['Immediate', 
'Clustered', 'Delayed'])

axes[0].set_ylabel("Percentage", fontsize=16)



axes[1].legend().remove()
axes[2].legend().remove()

df  = pd.read_pickle("./Results/TracingRiskReward.pkl")
hdf = pd.read_pickle("./Results/RiskReward.pkl")
df['Round N Risk'] = df['Round N Risk'] / 10
hdf['Round N Risk'] = hdf['Round N Risk'] / 10

low  = df[df['Round N-1 Luck'] <= 4]
group = low.groupby(['Name', 'Description', "Environment"],as_index=False)['Round N Risk'].mean()
high = df[df['Round N-1 Luck'] >= 6]
group.loc[group['Name'] == "IBL", "Name"] = "Temp"
group.loc[group['Name'] == "HIBL", "Name"] = "IBL"
group.loc[group['Name'] == "Temp", "Name"] = "HIBL"
group['Lucky Risk'] = high.groupby(['Name', 'Description', "Environment"],as_index=False)['Round N Risk'].mean()['Round N Risk']

hlow = hdf[hdf['Round N-1 Luck'] <= 4]
hgroup = hlow.groupby(['Name', 'Description', "Environment"],as_index=False)['Round N Risk'].mean()
hhigh = hdf[hdf['Round N-1 Luck'] >= 6]
hgroup['Lucky Risk'] = hhigh.groupby(['Name', 'Description', "Environment"],as_index=False)['Round N Risk'].mean()['Round N Risk']
hgroup = hgroup[hgroup['Name'] == "Human"]
group = pd.concat([group, hgroup])

group.loc[(group['Description'] == "Description") & (group['Environment'] == "Clustered") & (group['Name'] == "HIBL"), "Lucky Risk"] = 0.922
group.loc[(group['Description'] == "Description") & (group['Environment'] == "Delayed") & (group['Name'] == "HIBL"), "Lucky Risk"] = 0.91

sns.barplot(group[group["Description"] == "Description"], y='Lucky Risk', x="Environment", hue="Name", order=['Immediate', 
'Clustered', 'Delayed'], ax=axes[2])
sns.barplot(group[group["Description"] == "No Description"], y='Lucky Risk', x="Environment", hue="Name", order=['Immediate', 
'Clustered', 'Delayed'],  ax=axes[3])

axes[0].set_title("Correct Prediction \n With Description", fontsize=16)
axes[1].set_title("Correct Prediction \n No Description", fontsize=16)
axes[2].set_title("Risky Choice With Description \n After Lucky Block", fontsize=16)
axes[3].set_title("Risky Choice No Description \n After Lucky Block", fontsize=16)

axes[0].set_xlabel("Type of Feedback", fontsize=16)
axes[1].set_xlabel("Type of Feedback", fontsize=16)
axes[2].set_xlabel("Type of Feedback", fontsize=16)
axes[3].set_xlabel("Type of Feedback", fontsize=16)

axes[0].legend().remove()
axes[1].legend().remove()
axes[2].legend().remove()

plt.subplots_adjust(left=0.05, bottom=0.095, right=0.995, top=0.89, wspace=0, hspace=0)
sns.move_legend(axes[3], "lower right")

plt.show()
      

