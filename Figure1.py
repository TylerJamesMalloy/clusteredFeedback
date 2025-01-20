import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
import numpy as np


df = pd.read_pickle("Results/Simulations.pkl")
figure, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(15, 5))
palette = sns.color_palette('muted').as_hex()
palette = [palette[1], palette[0]]

human = df[df['Name'] == 'Human']
ibl = df[df['Name'] == 'IBL']
hibl = df[df['Name'] == 'HIBL']

#print(human.groupby(['Environment', 'Description'], as_index=False)['Risky'].mean()['Risky'].to_numpy())
print(ibl.groupby(['Environment', 'Description'], as_index=False)['Risky'].mean()['Risky'].to_numpy())
print(hibl.groupby(['Environment', 'Description'], as_index=False)['Risky'].mean()['Risky'].to_numpy())
groups = ['Immediate', 'Clustered', 'Delayed']
categories = ['Description', 'No Description']
values1 = [.42, .55, .57]
errors1 = [0.05, 0.05, 0.05]
values2 = [.74, .75, .76]
errors2 = [0.05, 0.05, 0.05]
bar_width = 0.4
x = np.arange(len(groups))

bars1 = axes[2].bar(x - bar_width/2, values1, bar_width, yerr=errors1, alpha=1,  color=palette[0], label='Description')
bars2 = axes[2].bar(x + bar_width/2, values2, bar_width, yerr=errors2, alpha=1,  color=palette[1], label='No Description')
axes[2].set_xticks(np.arange(len(groups)))  # Set tick positions
axes[2].set_xticklabels(groups)  # Set the new labels

sns.barplot(df[df['Name'] == 'HIBL'], x="Environment", y="Risky", hue="Description",  palette=palette, order=groups,  hue_order=["No Description", "Description"], ax=axes[1])
sns.barplot(df[df['Name'] == 'IBL'], x="Environment", y="Risky",  hue="Description", palette=palette, order=groups,  hue_order=["No Description", "Description"], ax=axes[0])
axes[1].legend().remove()

axes[0].legend(loc='upper left')
axes[0].set_ylabel("Probability of Risky Selection", fontsize=16)

axes[0].set_xlabel("Environment", fontsize=16)
axes[1].set_xlabel("Environment", fontsize=16)
axes[2].set_xlabel("Environment", fontsize=16)

axes[0].set_title("IBL Simulation", fontsize=20)
axes[1].set_title("Hierarchical IBL Simulation", fontsize=20)
axes[2].set_title("Human Behavior", fontsize=20)

for ns, bars in zip([[99, 100, 88], [101, 104, 94]], [bars1, bars2]):
    for idx, (n, bar) in enumerate(zip(ns, bars)):
        if(idx > 5): continue
        height = np.round((bar.get_height()*100),1)
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,  # X position
            0.05,  # Y position
            str(height) + "%",  # Text to display
            ha='center', va='bottom', fontsize=12, color='black'  # Alignment and style
        )
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,  # X position
            0.01,  # Y position
            "n=" + str(n),  # Text to display (formatted to 1 decimal place)
            ha='center', va='bottom', fontsize=12, color='black'  # Alignment and style
        )

for ax in [axes[0], axes[1]]:
    for idx, bar in enumerate(ax.patches):
        if(idx > 5): continue 
        height = np.round((bar.get_height()*100),1)
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # X position
            0.05,  # Y position
            str(height) + "%",  # Text to display (formatted to 1 decimal place)
            ha='center', va='bottom', fontsize=12, color='black'  # Alignment and style
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # X position
            0.01,  # Y position
            "n=1000",  # Text to display (formatted to 1 decimal place)
            ha='center', va='bottom', fontsize=10, color='black'  # Alignment and style
        )

axes[0].tick_params(axis='x', labelsize=14)  # Set size to 14
axes[1].tick_params(axis='x', labelsize=14)  # Set size to 14
axes[2].tick_params(axis='x', labelsize=14)  # Set size to 14
axes[0].legend(fontsize=12)  # Set font size to 12

plt.subplots_adjust(left=0.05, bottom=0.11, right=0.995, top=0.925, wspace=0)

plt.show()