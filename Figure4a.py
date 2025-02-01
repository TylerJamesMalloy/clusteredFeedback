import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
import numpy as np
#des = pd.read_pickle("Results/GapMeans.pkl")
df = pd.read_pickle("./Results/ModelTracing.pkl")

"""
Index(['Environment', 'Description', 'Name', 'Agent ID', 'Timestep',
    'True Reward', 'Observed Reward', 'Correct Prediction', 'Model Risky',
    'Human Risky', 'Human Reward', 'HIBL'],
    dtype='object')
"""
df.loc[df['Environment'] == 'Delayed', 'Environment'] = 'Aggregated'
print(df.columns)
df['Correct Prediction'] *= 100
fix, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4) )

print(df)

sns.barplot(df, x='Name', y="Correct Prediction", hue="Environment", hue_order=["Immediate", "Clustered", "Aggregated"]) 

ax.set_title("Model Tracing Percent Accuracy \n By Feedback and Model Type", fontsize=18)
ax.set_ylabel("Percent Accuracy in\nChoice Selection Prediction", fontsize=16)

ax.set_ylim(50,110)
ax.set_xlabel("Model Type", fontsize=16)
plt.legend(title='Type of Feedback')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.yticks([50, 75, 100], ['50%', '75%', '100%'])
sns.move_legend(ax, "upper left")
sns.move_legend(ax, "upper left", ncol=3)
plt.subplots_adjust(left=0.15, right=0.98, bottom=0.15, top=0.85, wspace=0, hspace=0)

plt.show()