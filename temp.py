import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
import numpy as np

sdf = pd.read_pickle("./Results/Simulations.pkl")
sdf = sdf[sdf['Name'] != 'Human']
"""
Index(['Environment', 'Description', 'Name', 'Agent ID', 'Timestep', 'Reward',
       'Risky'],
      dtype='object')
"""

fix, axes = plt.subplots(nrows=1, ncols=3, sharey=True)
group = sdf.groupby(['Environment', 'Description', 'Name'], as_index=False)['Risky'].mean()

des = group[group['Description'] == 'Description']
exp = group[group['Description'] == 'No Description']

# Immediate Gap: 32, Cumulative Gap: 23, Blocked Gap: 18
# Clustering feedback reduces the likelihood of switching after observing a loss by 18.45% (p < 0.01). 

des['Gap'] = (des['Risky'].to_numpy() - exp['Risky'].to_numpy()) * 100
des['Experiment'] = 'Experiment 1'
human = pd.DataFrame([
    {"Environment": "Delayed",   "Experiment": "Experiment 1", "Name":"Human", "Gap":18},
    {"Environment": "Clustered", "Experiment": "Experiment 1", "Name":"Human", "Gap":23},
    {"Environment": "Immediate", "Experiment": "Experiment 1", "Name":"Human", "Gap":32},
    {"Environment": "Delayed",   "Experiment": "Experiment 2 Likely Win", "Name":"Human", "Gap":1},
    {"Environment": "Clustered", "Experiment": "Experiment 2 Likely Win", "Name":"Human", "Gap":2},
    {"Environment": "Immediate", "Experiment": "Experiment 2 Likely Win", "Name":"Human", "Gap":1},
    {"Environment": "Delayed",   "Experiment": "Experiment 2 Likely Loss", "Name":"Human", "Gap":2},
    {"Environment": "Clustered", "Experiment": "Experiment 2 Likely Loss", "Name":"Human", "Gap":2},
    {"Environment": "Immediate", "Experiment": "Experiment 2 Likely Loss", "Name":"Human", "Gap":21},
    {"Environment": "Delayed",   "Experiment": "Experiment 2 Likely Win", "Name":"IBL", "Gap":1},
    {"Environment": "Clustered", "Experiment": "Experiment 2 Likely Win", "Name":"IBL", "Gap":2},
    {"Environment": "Immediate", "Experiment": "Experiment 2 Likely Win", "Name":"IBL", "Gap":1},
    {"Environment": "Delayed",   "Experiment": "Experiment 2 Likely Loss", "Name":"IBL", "Gap":2},
    {"Environment": "Clustered", "Experiment": "Experiment 2 Likely Loss", "Name":"IBL", "Gap":2},
    {"Environment": "Immediate", "Experiment": "Experiment 2 Likely Loss", "Name":"IBL", "Gap":21},
    {"Environment": "Delayed",   "Experiment": "Experiment 2 Likely Win", "Name":"HIBL", "Gap":1},
    {"Environment": "Clustered", "Experiment": "Experiment 2 Likely Win", "Name":"HIBL", "Gap":2},
    {"Environment": "Immediate", "Experiment": "Experiment 2 Likely Win", "Name":"HIBL", "Gap":1},
    {"Environment": "Delayed",   "Experiment": "Experiment 2 Likely Loss", "Name":"HIBL", "Gap":2},
    {"Environment": "Clustered", "Experiment": "Experiment 2 Likely Loss", "Name":"HIBL", "Gap":2},
    {"Environment": "Immediate", "Experiment": "Experiment 2 Likely Loss", "Name":"HIBL", "Gap":21},
])
des = pd.concat([human, des])

exp1 = des[des['Experiment'] == 'Experiment 1']
exp2 = des[des['Experiment'] == 'Experiment 2 Likely Win']
exp3 = des[des['Experiment'] == 'Experiment 2 Likely Loss']

sns.barplot(exp1, x='Name', y="Gap", hue="Environment", hue_order=["Immediate", "Clustered", "Delayed"], ax=axes[0])
sns.barplot(exp2, x='Name', y="Gap", hue="Environment", hue_order=["Immediate", "Clustered", "Delayed"], ax=axes[1])
sns.barplot(exp3, x='Name', y="Gap", hue="Environment", hue_order=["Immediate", "Clustered", "Delayed"], ax=axes[2])

axes[0].set_title("Experiment 1 Description-Experience Gap \n by Timing of Feedback")
axes[1].set_title("Experiment 2 Description-Experience Gap \n by Timing of Feedback (Likely Win)")
axes[2].set_title("Experiment 2 Description-Experience Gap \n by Timing of Feedback (Likely Loss)")

axes[0].set_ylabel("Description-Experience Gap (Percentage)", fontsize=14)

axes[0].set_xlabel("Model Type", fontsize=14)
axes[1].set_xlabel("Model Type", fontsize=14)
axes[2].set_xlabel("Model Type", fontsize=14)

axes[0].legend().remove()
axes[1].legend().remove()

plt.show()