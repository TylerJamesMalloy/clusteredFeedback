import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
import numpy as np

des = pd.read_pickle("./Results/SimulationGapMeans.pkl")

fix, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4) )


sns.barplot(des[des['Experiment'] == "Experiment 2"], x='Name', y="Gap", hue="Environment", hue_order=["Immediate", "Clustered", "Aggregated"])

ax.set_title("Description-Experience Gap\nby Timing of Feedback", fontsize=18)
ax.set_ylabel("Description-Experience Gap\nPercentage Point Difference", fontsize=16)
ax.set_xlabel("Human Participants or Model Type", fontsize=16)
plt.legend(title='Type of Feedback')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.set_ylim(0,30)
sns.move_legend(ax, "upper left")
sns.move_legend(ax, "upper left", ncol=3)
plt.subplots_adjust(left=0.15, right=0.98, bottom=0.15, top=0.85, wspace=0, hspace=0)

plt.show()