import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
import numpy as np
des = pd.read_pickle("Results/GapMeans.pkl")

fix, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4) )

sns.barplot(des[des['Experiment'] == "Experiment 1"], x='Name', y="Accuracy", hue="Environment", hue_order=["Immediate", "Clustered", "Aggregated"], ci=None)
#sns.barplot(des[des['Experiment'] == "Experiment 2"], x='Name', y="Gap", hue="Environment", hue_order=["Immediate", "Clustered", "Aggregated"], ax=axes[1])

ax.set_title("Model Tracing Accuracy By Feedback Type", fontsize=18)
ax.set_ylabel("Model Tracing Accuracy", fontsize=16)

ax.set_ylim(50,100)
ax.set_xlabel("Model Type", fontsize=16)
plt.legend(title='Type of Feedback')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.yticks([50, 70, 90], ['50%', '70%', '90%'])
sns.move_legend(ax, "upper left")
sns.move_legend(ax, "upper left", ncol=3)
plt.subplots_adjust(left=0.15, right=0.98, bottom=0.15, top=0.85, wspace=0, hspace=0)

plt.show()