import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
import numpy as np

mdf = pd.read_pickle("./Results/ModelTracing.pkl")
sdf = pd.read_pickle("./Results/Simulations.pkl")
sdf = sdf[sdf['Name'] != 'Human']
"""
Index(['Environment', 'Description', 'Name', 'Agent ID', 'Timestep', 'Reward',
       'Risky'],
      dtype='object')
"""

fix, axes = plt.subplots(nrows=1, ncols=2)
group = sdf.groupby(['Environment', 'Description', 'Name'], as_index=False)['Risky'].mean()

des = group[group['Description'] == 'Description']
exp = group[group['Description'] == 'No Description']

# Immediate Gap: 32, Cumulative Gap: 23, Blocked Gap: 18
# Clustering feedback reduces the likelihood of switching after observing a loss by 18.45% (p < 0.01). 

des['Gap'] = (des['Risky'].to_numpy() - exp['Risky'].to_numpy()) * 100
des['Experiment'] = 'Experiment 1'
print(des.columns)
human = pd.DataFrame([
    {"Environment": "Aggregated",   "Experiment": "Experiment 2", "Name":"Human", "Gap":18.5},
    {"Environment": "Clustered", "Experiment": "Experiment 2", "Name":"Human", "Gap":16},
    {"Environment": "Immediate", "Experiment": "Experiment 2", "Name":"Human", "Gap":3},
    {"Environment": "Aggregated",   "Experiment": "Experiment 2", "Name":"HIBL", "Gap":19},
    {"Environment": "Clustered", "Experiment": "Experiment 2", "Name":"HIBL", "Gap":17},
    {"Environment": "Immediate", "Experiment": "Experiment 2", "Name":"HIBL", "Gap":6},
    {"Environment": "Aggregated",   "Experiment": "Experiment 2", "Name":"IBL", "Gap":12},
    {"Environment": "Clustered", "Experiment": "Experiment 2", "Name":"IBL", "Gap":14},
    {"Environment": "Immediate", "Experiment": "Experiment 2", "Name":"IBL", "Gap":18},
])
des = pd.concat([human, des])
exp1  = pd.DataFrame([
    {"Environment": "Aggregated",   "Experiment": "Experiment 1", "Name":"HIBL", "Accuracy":81},
    {"Environment": "Clustered", "Experiment": "Experiment 1", "Name":"HIBL", "Accuracy":84},
    {"Environment": "Immediate", "Experiment": "Experiment 1", "Name":"HIBL", "Accuracy":72.5},
    {"Environment": "Aggregated",   "Experiment": "Experiment 1", "Name":"IBL", "Accuracy":67},
    {"Environment": "Clustered", "Experiment": "Experiment 1", "Name":"IBL", "Accuracy":69},
    {"Environment": "Immediate", "Experiment": "Experiment 1", "Name":"IBL", "Accuracy":80},
])



des = pd.concat([exp1, des])
des.to_pickle("Results/GapMeans.pkl")

sns.barplot(exp1, x='Name', y="Accuracy", hue="Environment", hue_order=["Immediate", "Clustered", "Aggregated"], ax=axes[0])
sns.barplot(des[des['Experiment'] == "Experiment 2"], x='Name', y="Gap", hue="Environment", hue_order=["Immediate", "Clustered", "Aggregated"], ax=axes[1])

axes[0].set_title("Model Tracing Accuracy \n By Feedback Type")
axes[1].set_title("Description-Experience Gap \n After Lucky Blocks")
axes[0].set_ylabel("Model Tracing Accuracy", fontsize=14)
axes[1].set_ylabel("Description-Experience Gap", fontsize=14)

axes[0].set_ylim(50,100)
axes[0].set_xlabel("Model Type", fontsize=14)
axes[1].set_xlabel("Model Type", fontsize=14)

axes[1].legend().remove()

plt.show()