import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import sys 
import argparse
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
import numpy as np
import random 
import tqdm

df = pd.read_pickle("./Simulations/Results.pkl")


palette = sns.color_palette(palette='tab10')
palette = [palette[1], palette[0]]
fig, axes = plt.subplots(nrows=2, ncols=2, sharey=True)
sns.barplot(df, x="Environment", y="Risky", hue="Description", palette=palette, hue_order=["No Description", "Description"], ax=axes[0][0])

axes[0][0].set_title("IBL")
axes[0][0].set_ylim(0,1)
axes[0][0].set_ylabel("Choice of the Risky Option")
plt.show()
if(args.trace):
    df.to_pickle("./ModelTracing/Results.pkl")
else:
    df.to_pickle("./Simulations/Results.pkl")