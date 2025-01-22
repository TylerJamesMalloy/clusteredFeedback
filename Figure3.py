
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
  
fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(8, 3))

df = pd.read_pickle("./Results/ModelTracing.pkl")

group = df.groupby(['Environment', 'Description', 'Name', 'Agent ID'], as_index=False)["Correct Prediction"].mean()
immediate = group[group['Environment'] == "Immediate"]
clustered = group[group['Environment'] == "Clustered"]
delayed = group[group['Environment'] == "Delayed"]

sns.barplot(data=immediate, x="Description", y="Correct Prediction", hue="Name", hue_order=["HIBL", "IBL"], ax=axes[0])
sns.barplot(data=clustered, x="Description", y="Correct Prediction", hue="Name",  ax=axes[1])
sns.barplot(data=delayed, x="Description", y="Correct Prediction", hue="Name", hue_order=["HIBL", "IBL"], ax=axes[2])

axes[0].set_ylabel("Correct Prediction", fontsize=16)

axes[0].set_title("Immediate Feedback", fontsize=16)
axes[1].set_title("Clustered Feedback", fontsize=16)
axes[2].set_title("Delayed Feedback", fontsize=16)

axes[0].set_xlabel("Description", fontsize=16)
axes[1].set_xlabel("Description", fontsize=16)
axes[2].set_xlabel("Description", fontsize=16)

axes[1].legend().remove()
axes[2].legend().remove()

axes[0].set_ylim(0.5,0.9)
axes[1].set_ylim(0.5,0.9)
axes[2].set_ylim(0.5,0.9)

plt.subplots_adjust(left=0.1, bottom=0.095, right=0.995, top=0.915, wspace=0, hspace=0)

plt.show()
      

