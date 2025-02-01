import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
import numpy as np

Data = pd.read_pickle("./Results/Simulations.pkl")
columns = ['Environment', 'Description', 'Name', 'Agent ID', 'Lucky Gap', 'Unlucky Gap', 'Safe Gap', 'Risky Gap']
bdf = pd.DataFrame([], columns=columns)

for Environment in Data['Environment'].unique():
    EnvironmentData = Data[Data['Environment'] == Environment]
    DescriptionRiskAfterLucky = []
    ExperienceRiskAfterLucky  = []
    DescriptionRiskAfterUnlucky = []
    ExperienceRiskAfterUnlucky  = []
    DescriptionRiskAfterSafe = []
    ExperienceRiskAfterSafe  = []
    DescriptionRiskAfterUnsafe = []
    ExperienceRiskAfterUnsafe  = []

    for Description in EnvironmentData['Description'].unique():
        DescriptionData = EnvironmentData[EnvironmentData['Description'] == Description]
        for Name in DescriptionData['Name'].unique():
            NameData = DescriptionData[DescriptionData['Name'] == Name]
            LuckyGaps = [] 
            UnluckyGaps = [] 
            SafeGaps = [] 
            RiskyGaps = [] 

            for AgentID in NameData['Agent ID'].unique():
                AgentData = NameData[NameData['Agent ID'] == AgentID]
                prevBlockLucky = None 
                prevBlockUnlucky  = None 
                prevBlockSafe = None 
                prevBlockRisky = None 

                for Block, Range in enumerate([[0,9],[10,19],[20,29],[30,39],[40,49],[50,59],[60,69],[70,79],[80,89],[90,99]]):
                    BlockData = AgentData[(AgentData['Timestep'] > Range[0]) & (AgentData['Timestep'] < Range[1])]
                    BlockLucky = BlockData['Lucky'].sum()
                    BlockUnlucky = BlockData['Unlucky'].sum()
                    BlockSafe = 10 - BlockData['Risky'].sum()
                    BlockRisky = BlockData['Risky'].sum()
                    
                    if(prevBlockLucky is not None):
                        if(prevBlockLucky >= 6):
                            print(prevBlockLucky)

                            assert(False)
                    
                    prevBlockLucky = BlockLucky
                    prevBlockUnlucky = BlockUnlucky
                    prevBlockSafe = BlockSafe
                    prevBlockRisky = BlockRisky


    DescriptionRiskAfterLucky = np.mean(DescriptionRiskAfterLucky)
    ExperienceRiskAfterLucky  = np.mean(ExperienceRiskAfterLucky)
    DescriptionRiskAfterUnlucky = np.mean(DescriptionRiskAfterUnlucky)
    ExperienceRiskAfterUnlucky  = np.mean(ExperienceRiskAfterUnlucky)
    DescriptionRiskAfterSafe = np.mean(DescriptionRiskAfterSafe)
    ExperienceRiskAfterSafe  = np.mean(ExperienceRiskAfterSafe)
    DescriptionRiskAfterUnsafe = np.mean(DescriptionRiskAfterUnsafe)
    ExperienceRiskAfterUnsafe  = np.mean(ExperienceRiskAfterUnsafe)

assert(False)
fix, axes = plt.subplots(nrows=1, ncols=2, sharey=True)


sns.barplot(des[des['Experiment'] == "Experiment 1"], x='Name', y="Gap", hue="Environment", hue_order=["Immediate", "Clustered", "Aggregated"], ax=axes[0])
sns.barplot(des[des['Experiment'] == "Experiment 2"], x='Name', y="Gap", hue="Environment", hue_order=["Immediate", "Clustered", "Aggregated"], ax=axes[1])

axes[0].set_title("Description-Experience Gap \n by Timing of Feedback")
axes[1].set_title("Description-Experience Gap \n After Lucky Blocks")
axes[0].set_ylabel("Description-Experience Gap", fontsize=14)

axes[0].set_xlabel("Model Type", fontsize=14)
axes[1].set_xlabel("Model Type", fontsize=14)

axes[0].legend().remove()

plt.show()