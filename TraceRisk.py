

import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit

df = pd.read_pickle("./Results/ModelTracing.pkl")
"""
Index(['Environment', 'Description', 'Name', 'Agent ID', 'Timestep',
       'True Reward', 'Observed Reward', 'Correct Prediction', 'Model Risky',
       'Human Risky', 'Human Reward', 'HIBL'],
      dtype='object')
"""

df['Lucky'] = None
df['Unlucky'] = None
df.loc[df['True Reward'] == 0, 'Unlucky'] = 1
df.loc[df['True Reward'] == 10, 'Unlucky'] = 0
df.loc[df['True Reward'] == 10, 'Lucky'] = 1
df.loc[df['True Reward'] == 0, 'Lucky'] = 0

begining = df[df['Timestep'] < 20]
end      = df[df['Timestep'] > 80]

group = end.groupby(['Environment', 'Description', 'Name', 'Agent ID'], as_index=False)['Model Risky'].mean()
lucky = begining.groupby(['Environment', 'Description', 'Name', 'Agent ID'], as_index=False)['Lucky'].mean()['Lucky'].to_numpy()
unlucky = begining.groupby(['Environment', 'Description', 'Name', 'Agent ID'], as_index=False)['Unlucky'].mean()['Unlucky'].to_numpy()
#lucky = np.round(lucky.astype(float), 1)
#unlucky = np.round(unlucky.astype(float), 1)
group['Lucky'] = lucky
group['Unlucky'] = unlucky 

group['Lucky'] = group['Lucky'].astype(float)
group['Unlucky'] = group['Unlucky'].astype(float)
group['Model Risky'] = group['Model Risky'].astype(float)

fig, axes = plt.subplots(ncols=6, sharey=True, figsize=(15, 5))

imm = group[group['Environment'] == "Clustered"]
imm = imm[imm['Name'] == "HIBL"]
print("Clustered")
print(imm[imm['Lucky'] < 1/3]['Model Risky'].mean())
print(imm[imm['Lucky'] > 2/3]['Model Risky'].mean())