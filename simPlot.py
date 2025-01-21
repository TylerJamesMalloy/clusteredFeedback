import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

ldf = pd.read_pickle("./Results/RiskReward.pkl")

fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(15, 6))
humanDesc = ldf[(ldf['Name'] == "Human") & (ldf['Description'] == 'Description')]
hiblDesc = ldf[(ldf['Name'] == "HIBL") & (ldf['Description'] == 'Description')]
iblDesc = ldf[(ldf['Name'] == "IBL") & (ldf['Description'] == 'Description')]

humanDesc = humanDesc.groupby(["Round N-1 Luck"], as_index=False)['Round N Risk'].mean()
hiblDesc = hiblDesc.groupby(["Round N-1 Luck"], as_index=False)['Round N Risk'].mean()
iblDesc = iblDesc.groupby(["Round N-1 Luck"], as_index=False)['Round N Risk'].mean()

humanDesc = humanDesc.groupby(["Round N-1 Luck"], as_index=False)['Round N Risk'].mean()
hiblDesc = hiblDesc.groupby(["Round N-1 Luck"], as_index=False)['Round N Risk'].mean()
iblDesc = iblDesc.groupby(["Round N-1 Luck"], as_index=False)['Round N Risk'].mean()

sns.regplot(humanDesc,  x="Round N-1 Luck", y="Round N Risk", label="Human", order=2, ax=axes[0])
sns.regplot(hiblDesc,   x="Round N-1 Luck", y="Round N Risk", label="HIBL", order=2, ax=axes[0])
sns.regplot(iblDesc,    x="Round N-1 Luck", y="Round N Risk", label="IBL", order=1, ax=axes[0])

humanNoDesc = ldf[(ldf['Name'] == "Human") & (ldf['Description'] == 'No Description')]
hiblNoDesc = ldf[(ldf['Name'] == "HIBL") & (ldf['Description'] == 'No Description')]
iblNoDesc = ldf[(ldf['Name'] == "IBL") & (ldf['Description'] == 'No Description')]

humanNoDesc = humanNoDesc.groupby(["Round N-1 Luck"], as_index=False)['Round N Risk'].mean()
hiblNoDesc = hiblNoDesc.groupby(["Round N-1 Luck"], as_index=False)['Round N Risk'].mean()
iblNoDesc = iblNoDesc.groupby(["Round N-1 Luck"], as_index=False)['Round N Risk'].mean()

sns.regplot(humanNoDesc, x="Round N-1 Luck", y="Round N Risk", label="Human", ax=axes[1])
sns.regplot(hiblNoDesc, x="Round N-1 Luck", y="Round N Risk", label="HIBL", ax=axes[1])
sns.regplot(iblNoDesc, x="Round N-1 Luck", y="Round N Risk", label="IBL", order=2, ax=axes[1])

axes[0].set_title("Description")
axes[1].set_title("No Description")

plt.show()


X = humanDesc["Round N Risk"].to_numpy().reshape(-1, 1)
y = humanDesc["Round N-1 Luck"].to_numpy().reshape(-1, 1)
clf = LinearRegression().fit(X, y)
humanDescQuadratic= clf.score(X,y)

X = humanDesc["Round N Risk"].to_numpy().reshape(-1, 1)
y = humanDesc["Round N-1 Luck"].to_numpy().reshape(-1, 1)
clf = LinearRegression().fit(X, y)
humanDescLinear = clf.score(X,y)

X = hiblDesc["Round N Risk"].to_numpy().reshape(-1, 1)
y = hiblDesc["Round N-1 Luck"].to_numpy().reshape(-1, 1)
clf = LinearRegression().fit(X, y)
hiblDescQuadratic= clf.score(X,y)

X = hiblDesc["Round N Risk"].to_numpy().reshape(-1, 1)
y = hiblDesc["Round N-1 Luck"].to_numpy().reshape(-1, 1)
clf = LinearRegression().fit(X, y)
hiblDescLinear = clf.score(X,y)

X = iblDesc["Round N Risk"].to_numpy().reshape(-1, 1)
y = iblDesc["Round N-1 Luck"].to_numpy().reshape(-1, 1)
clf = LinearRegression().fit(X, y)
iblDescQuadratic= clf.score(X,y)

X = iblDesc["Round N Risk"].to_numpy().reshape(-1, 1)
y = iblDesc["Round N-1 Luck"].to_numpy().reshape(-1, 1)
clf = LinearRegression().fit(X, y)
iblDescLinear = clf.score(X,y)

X = humanDesc["Round N Risk"].to_numpy().reshape(-1, 1)
y = humanDesc["Round N-1 Luck"].to_numpy().reshape(-1, 1)
clf = LinearRegression().fit(X, y)
humanDescQuadratic= clf.score(X,y)

X = humanDesc["Round N Risk"].to_numpy().reshape(-1, 1)
y = humanDesc["Round N-1 Luck"].to_numpy().reshape(-1, 1)
clf = LinearRegression().fit(X, y)
humanDescLinear = clf.score(X,y)

X = hiblDesc["Round N Risk"].to_numpy().reshape(-1, 1)
y = hiblDesc["Round N-1 Luck"].to_numpy().reshape(-1, 1)
clf = LinearRegression().fit(X, y)
hiblDescQuadratic= clf.score(X,y)

X = hiblDesc["Round N Risk"].to_numpy().reshape(-1, 1)
y = hiblDesc["Round N-1 Luck"].to_numpy().reshape(-1, 1)
clf = LinearRegression().fit(X, y)
hiblDescLinear = clf.score(X,y)

X = iblDesc["Round N Risk"].to_numpy().reshape(-1, 1)
y = iblDesc["Round N-1 Luck"].to_numpy().reshape(-1, 1)
clf = LinearRegression().fit(X, y)
iblDescQuadratic= clf.score(X,y)

X = iblDesc["Round N Risk"].to_numpy().reshape(-1, 1)
y = iblDesc["Round N-1 Luck"].to_numpy().reshape(-1, 1)
clf = LinearRegression().fit(X, y)
iblDescLinear = clf.score(X,y)

print("No Description: Linear: Human: ", humanNoDescLinear, " HIBL ", hiblNoDescLinear, " IBL ", iblNoDescLinear)
print("No Description: Quadratic: Human: ", humanNoDescQuadratic, " HIBL ", hiblNoDescQuadratic, " IBL ", iblDescNoDescQuadratic)

NoDescLinear = [humanNoDescLinear, hiblNoDescLinear, iblNoDescLinear]
NoDescQuadratic = [humanNoDescQuadratic, hiblNoDescQuadratic, iblNoDescQuadratic]


print("Description: Linear: Human: ", humanDescLinear, " HIBL ", hiblDescLinear, " IBL ", iblDescLinear)
print("Description: Quadratic: Human: ", humanDescQuadratic, " HIBL ", hiblDescQuadratic, " IBL ", iblDescQuadratic)

DescLinear = [humanDescLinear, hiblDescLinear, iblDescLinear]
DescQuadratic = [humanDescLinear, hiblDescLinear, iblDescLinear]
