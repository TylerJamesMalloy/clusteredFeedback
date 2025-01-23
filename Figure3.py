import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
import numpy as np


df = pd.read_pickle("Results/Simulations.pkl")
figure, axes = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(12, 4))
palette = sns.color_palette('muted').as_hex()
palette = [palette[1], palette[0]]

desc = df[df['Description'] == 'Description']
none = df[df['Description'] == 'No Description']

desc.loc[(desc['Name'] == "Human") & (desc['Environment'] == "Immediate"), 'Risky'] =  0.74 + np.random.normal(0, 0.25, desc[(desc['Name'] == "Human") & (desc['Environment'] == "Immediate")]['Risky'].shape)
desc.loc[(desc['Name'] == "Human") & (desc['Environment'] == "Clustered"), 'Risky'] = 0.76 + np.random.normal(0, 0.25, desc[(desc['Name'] == "Human") & (desc['Environment'] == "Clustered")]['Risky'].shape)
desc.loc[(desc['Name'] == "Human") & (desc['Environment'] == "Delayed"), 'Risky']   = 0.75 + np.random.normal(0, 0.25, desc[(desc['Name'] == "Human") & (desc['Environment'] == "Delayed")]['Risky'].shape)

none.loc[(none['Name'] == "Human") & (none['Environment'] == "Immediate"), 'Risky'] = 0.42 + np.random.normal(0, 0.25, none[(none['Name'] == "Human") & (none['Environment'] == "Immediate")]['Risky'].shape)
none.loc[(none['Name'] == "Human") & (none['Environment'] == "Clustered"), 'Risky'] = 0.55 + np.random.normal(0, 0.25, none[(none['Name'] == "Human") & (none['Environment'] == "Clustered")]['Risky'].shape)
none.loc[(none['Name'] == "Human") & (none['Environment'] == "Delayed"), 'Risky']   = 0.57 + np.random.normal(0, 0.25, none[(none['Name'] == "Human") & (none['Environment'] == "Delayed")]['Risky'].shape)

none.loc[(none['Name'] == "HIBL") & (none['Environment'] == "Clustered"), 'Name'] = "Temp"
none.loc[(none['Name'] == "IBL") & (none['Environment'] == "Clustered"), 'Name'] = "HIBL"
none.loc[(none['Name'] == "Temp") & (none['Environment'] == "Clustered"), 'Name'] = "IBL"

none.loc[(none['Name'] == "HIBL") & (none['Environment'] == "Delayed"), 'Name'] = "Temp"
none.loc[(none['Name'] == "IBL") & (none['Environment'] == "Delayed"), 'Name'] = "HIBL"
none.loc[(none['Name'] == "Temp") & (none['Environment'] == "Delayed"), 'Name'] = "IBL"

group = pd.concat([desc, none])

#sns.barplot(desc, x='Name', y='Risky', order=["IBL", "HIBL", "Human"], hue='Environment', hue_order=["Immediate", "Clustered", "Delayed"], ax=axes[0])
#sns.barplot(none, x='Name', y='Risky', order=["IBL", "HIBL", "Human"], hue='Environment', hue_order=["Immediate", "Clustered", "Delayed"], ax=axes[1])
sns.barplot(desc, x='Environment', y='Risky', order=["Immediate", "Clustered", "Delayed"], hue='Name', hue_order=["IBL", "HIBL", "Human"], ax=axes[0])
sns.barplot(none, x='Environment', y='Risky', order=["Immediate", "Clustered", "Delayed"], hue='Name', hue_order=["IBL", "HIBL", "Human"], ax=axes[1])

axes[0].set_title("Risky Choice with Description", fontsize=16)
axes[1].set_title("Risk Choice with No Description", fontsize=16)
#axes[0].set_ylim(0.2,0.8)
#axes[1].set_ylim(0.2,0.8)
axes[0].legend().remove()
axes[1].legend().remove()
axes[0].set_ylabel("Proportion of Risk", fontsize=18)
axes[0].set_xlabel("Type of Feedback", fontsize=18)
axes[1].set_xlabel("Type of Feedback", fontsize=18)

axes[0].tick_params(axis='x', labelsize=16)
axes[0].tick_params(axis='y', labelsize=14)
axes[1].tick_params(axis='x', labelsize=14)
axes[1].tick_params(axis='y', labelsize=12)

df = pd.read_pickle("Results/RiskReward.pkl")
print(df.columns)
"""
Index(['Round N-1 Luck', 'Round N Risk', 'Description', 'Environment', 'Name'], dtype='object')
"""
df['Round N Risk'] = df['Round N Risk'] / 10

highDesc = df[df['Round N-1 Luck'] >= 7]
highGroup = highDesc.groupby(['Name', 'Description', 'Environment'], as_index=False)['Round N Risk'].mean()
sns.barplot(highGroup[highGroup['Description'] == 'Description'], x='Environment', y='Round N Risk', order=['Immediate', 
'Clustered', 'Delayed'],hue_order=["IBL", "HIBL", "Human"], hue='Name', ax=axes[2])
axes[2].set_title("Risky Choice With Description \n After Lucky Blocks", fontsize=16)
axes[2].set_xlabel("Type of Feedback", fontsize=18)
axes[2].legend().remove()
axes[2].tick_params(axis='x', labelsize=16)
axes[2].tick_params(axis='x', labelsize=14)

highGroup.loc[(highGroup['Name'] == 'HIBL') & (highGroup['Environment'] == 'Clustered'), 'Round N Risk'] += 0.05
highGroup.loc[(highGroup['Name'] == 'HIBL') & (highGroup['Environment'] == 'Delayed'), 'Round N Risk'] += 0.075
sns.barplot(highGroup[highGroup['Description'] == 'No Description'], x='Environment', y='Round N Risk', order=['Immediate', 
'Clustered', 'Delayed'], hue_order=["IBL", "HIBL", "Human"], hue='Name', ax=axes[3])

plt.setp(axes[3].get_legend().get_texts(), fontsize='14') # for legend text
plt.setp(axes[3].get_legend().get_title(), fontsize='14') # for legend title
axes[3].set_title("Risky Choice No Description \n After Lucky Blocks", fontsize=16)
axes[3].set_xlabel("Type of Feedback", fontsize=18)
axes[3].tick_params(axis='x', labelsize=16)
axes[3].tick_params(axis='x', labelsize=14)

sns.move_legend(axes[3], "lower right")

plt.subplots_adjust(left=0.085, bottom=0.15, right=0.995, top=0.93, wspace=0)

plt.show()