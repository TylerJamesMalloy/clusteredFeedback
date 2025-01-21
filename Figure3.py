
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

fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(8, 6))

group = df.groupby(['Environment', 'Description', 'Name', 'Agent ID'], as_index=False)["Correct Prediction"].mean()
immediate = group[group['Environment'] == "Immediate"]
clustered = group[group['Environment'] == "Clustered"]
delayed = group[group['Environment'] == "Delayed"]

sns.violinplot(data=immediate, x="Description", y="Correct Prediction", hue="Name", ax=axes[0])
sns.violinplot(data=clustered, x="Description", y="Correct Prediction", hue="Name", ax=axes[1])
sns.violinplot(data=delayed, x="Description", y="Correct Prediction", hue="Name", ax=axes[2])

axes[0].set_ylabel("Probability of Correct Prediction", fontsize=16)

axes[0].set_title("Immediate Feedback", fontsize=16)
axes[1].set_title("Clustered Feedback", fontsize=16)
axes[2].set_title("Delayed Feedback", fontsize=16)

axes[0].set_xlabel("Description Present", fontsize=16)
axes[1].set_xlabel("Description Present", fontsize=16)
axes[2].set_xlabel("Description Present", fontsize=16)

axes[1].legend().remove()
axes[2].legend().remove()

plt.subplots_adjust(left=0.08, bottom=0.085, right=0.995, top=0.875, wspace=0, hspace=0)

plt.show()
      

