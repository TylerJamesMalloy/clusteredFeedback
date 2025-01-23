import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
import tqdm

df = pd.read_pickle("./Results/ModelTracing.pkl")
columns = ["Round N Risk", "Round N-1 Luck", "Name", "Agent ID", "Environment", "Description"]
ldf = pd.DataFrame([])

"""
for agent in tqdm.tqdm(df['Agent ID'].unique()):
    adf = df[(df['Agent ID'] == agent)]
    for name in df['Name'].unique():
        ndf = adf[(adf['Name'] == name)]
        Round_N1_Luck = 0 
        Round_N_Risk = 0 
        Round_N_Luck = 0
        for ts in df['Timestep'].unique():
            tdf = ndf[(ndf['Timestep'] == ts)]
            if(len(tdf) != 1): continue 
            Round_N_Risk += 1 if tdf['Model Risky'].item() else 0
            Round_N_Luck += 1 if tdf['True Reward'].item() == 10 else 0
            if(ts % 10 == 0):
                if(ts > 20):
                    ld = pd.DataFrame([{"Round N Risk": Round_N_Risk, 
                           "Round N-1 Luck":Round_N1_Luck, 
                           "Name": name,
                           "Agent ID": agent,
                           "Environment":tdf['Environment'].unique()[0],
                           "Description":tdf['Description'].unique()[0],
                           }], columns=columns)
                    ldf = pd.concat([ldf, ld], ignore_index=True)
                
                Round_N1_Luck = Round_N_Luck
                Round_N_Luck = 0
                Round_N_Risk = 0

print(ldf['Round N Risk'].unique()) 
print(ldf['Round N-1 Luck'].unique())      

ldf.to_pickle("./Results/TracingRiskReward.pkl")
"""
ldf = pd.read_pickle("./Results/TracingRiskReward.pkl")
hdf = pd.read_pickle("./Results/RiskReward.pkl")
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))
humanDesc = hdf[(hdf['Name'] == "Human") & (hdf['Description'] == 'Description')]
hiblDesc = ldf[(ldf['Name'] == "HIBL") & (ldf['Description'] == 'Description')]
iblDesc = ldf[(ldf['Name'] == "IBL") & (ldf['Description'] == 'Description')]


humanDesc = humanDesc.groupby(["Round N-1 Luck"], as_index=False)['Round N Risk'].mean()
hiblDesc = hiblDesc.groupby(["Round N-1 Luck"], as_index=False)['Round N Risk'].mean()
iblDesc = iblDesc.groupby(["Round N-1 Luck"], as_index=False)['Round N Risk'].mean()

humanDesc = humanDesc.groupby(["Round N-1 Luck"], as_index=False)['Round N Risk'].mean()
hiblDesc = hiblDesc.groupby(["Round N-1 Luck"], as_index=False)['Round N Risk'].mean()
iblDesc = iblDesc.groupby(["Round N-1 Luck"], as_index=False)['Round N Risk'].mean()

sns.regplot(humanDesc,  x="Round N-1 Luck", y="Round N Risk", label="Human", order=2, ax=axes[0], truncate=False)
sns.regplot(hiblDesc,   x="Round N-1 Luck", y="Round N Risk", label="HIBL", order=2, ax=axes[0], truncate=False)
sns.regplot(iblDesc,    x="Round N-1 Luck", y="Round N Risk", label="IBL", order=1, ax=axes[0], truncate=False)

humanNoDesc = hdf[(hdf['Name'] == "Human") & (hdf['Description'] == 'No Description')]
hiblNoDesc = ldf[(ldf['Name'] == "HIBL") & (ldf['Description'] == 'No Description')]
iblNoDesc = ldf[(ldf['Name'] == "IBL") & (ldf['Description'] == 'No Description')]

humanNoDesc = humanNoDesc.groupby(["Round N-1 Luck"], as_index=False)['Round N Risk'].mean()
hiblNoDesc = hiblNoDesc.groupby(["Round N-1 Luck"], as_index=False)['Round N Risk'].mean()
iblNoDesc = iblNoDesc.groupby(["Round N-1 Luck"], as_index=False)['Round N Risk'].mean()

sns.regplot(humanNoDesc, x="Round N-1 Luck", y="Round N Risk", label="Human", ax=axes[1], truncate=False)
sns.regplot(hiblNoDesc, x="Round N-1 Luck", y="Round N Risk", label="HIBL", ax=axes[1], truncate=False)
sns.regplot(iblNoDesc, x="Round N-1 Luck", y="Round N Risk", label="IBL", order=2, ax=axes[1], truncate=False)

cols = ["Model", "Description", "Regression Error"]
errs = pd.DataFrame([
    {"Model":"IBL", "Description": "No Description", "Regression Error": 0.76},
    {"Model":"IBL", "Description": "Description", "Regression Error": 0.86},
    {"Model":"HIBL", "Description": "No Description", "Regression Error": 0.56},
    {"Model":"HIBL", "Description": "Description", "Regression Error": 0.66},
], columns=cols)

sns.barplot(errs, x="Model", y="Regression Error", hue="Description", ax=axes[2])

#axes[0].set_xlim(0,10)
#axes[1].set_xlim(0,10)

#axes[0].set_ylim(0,10)
#axes[1].set_ylim(0,10)
axes[0].legend()

axes[0].set_title("Description")
axes[1].set_title("No Description")




plt.show()