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

from Models import  *
from Envrionments import * 
def Optimize(param, args):
    model = param['model']
    args.pretrainNo = param['pretrainNo']
    args.pretrainDesc = param['pretrainDesc']
    args.noise = param['noise']
    args.temperature = param['temperature']
    args.decay = param['decay']
    
    envirs = ['Immediate', 'Delayed', 'Clustered']
    #envirs = ['Immediate']
    descrs = ["Description", "No Description"]

    cols = ["Environment", "Description", "Name", "Agent ID", "Timestep", "Reward", "Risky"]
    df = pd.DataFrame([], columns=cols)
    for ident in range(args.agents):
        for envir in envirs:
            for descr in descrs:
                args.envir = envir 
                args.descr = descr
                args.ident = ident
                args.model = model
                args.param = param
                df = pd.concat([df, Train(args)], ignore_index=True)

    guess = df.groupby(['Environment', 'Description'], as_index=False)['Risky'].mean()['Risky'].to_numpy()
    # Clustered Description, Clustered No, Delayed Description, Delayed No, Immediate Description, Immediate No: [0.563 0.518 0.563 0.518 0.513 0.35 ]
    true = [0.563, 0.518, 0.563, 0.518, 0.513 , 0.350]

    return np.sum((np.array(true) - np.array(guess)) ** 2), df

"""
IBL Best parameters:  {'model': 'IBLAgent', 'pretrainNo': 0, 'pretrainDesc': 100, 'noise': 0.2, 'temperature': 0.5, 'decay': 0.3, 'error': np.float64(0.023640000000000015), 'df': 0}
HIBL Best parameters:  {'model': 'IBLAgent', 'pretrainNo': 0, 'pretrainDesc': 25, 'noise': 0.2, 'temperature': 0.5, 'decay': 0.1, 'error': np.float64(0.022615), 'df': 0}
"""
def Train(args):
    cols = ["Environment", "Description", "Name", "Agent ID", "Timestep", "True Reward", "Observed Reward", "Risky"]
    df = pd.DataFrame([], columns=cols)
    agent = getattr(sys.modules[__name__], args.model)(args)
    envir = getattr(sys.modules[__name__], args.envir)(args)
    agent.pretrain()
    for timestep in range(args.timesteps):
        observed_reward, true_reward, risky = envir.reward(agent.choose(options=envir.options()))
        agent.respond(observed_reward)
        df = pd.concat([df, pd.DataFrame([[envir.name, args.descr, agent.name, args.ident, timestep, true_reward, observed_reward, risky]], columns=cols)], ignore_index=True)
    return df

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    parser.add_argument("--dataPath", type=str, default="/HumanBehavior/study1.csv", help="Path to human decisions.")
    parser.add_argument("--trace", type=bool, default=False, help="Whether or not to trace the human decisions in dataPath")
    parser.add_argument("--agents", type=int, default=10, help="Num Agents")
    parser.add_argument("--window", type=int, default=10, help="Feedback window size")
    parser.add_argument("--risky", type=list, default=[[0,10],[0.5,0.5]], help="Probabilities for risky option (B).")
    parser.add_argument("--sure", type=int, default=4, help="Value of the sure option (A).")
    parser.add_argument("--pretrainNo", type=int, default=50, help="Pretraining steps for description environment.")
    parser.add_argument("--pretrainDesc", type=int, default=75, help="Pretraining steps for description environment.")
    parser.add_argument("--noise", type=float, default=0.25, help="IBL noise parameter")
    parser.add_argument("--temperature", type=float, default=0.5, help="IBL temperature parameter")
    parser.add_argument("--decay", type=float, default=0.25, help="IBL decay parameter")
    parser.add_argument("--timesteps", type=int, default=100, help="Num timesteps")
    args = parser.parse_args()
    
    cols = ["Environment", "Description", "Name", "Agent ID", "Timestep", "True Reward", "Observed Reward", "Risky"]
    df = pd.DataFrame([], columns=cols)

    #tqdm.tqdm()
    models = ["HIBLAgent"]
    pretrainNos = [0]
    pretrainDescs = [100, 200]
    noises = [0.1,  0.4]
    temperatures = [0.2, 0.7]
    decays = [0.2,  0.7]
    params = []
    pbar = tqdm.tqdm(total=(len(models) * len(pretrainNos) * len(pretrainDescs) * len(noises) * len(temperatures) * len(decays)))
    for model in models:
        for pretrainNo in pretrainNos:
            for pretrainDesc in pretrainDescs:
                for noise in noises:
                    for temperature in temperatures:
                        for decay in decays:
                            param = {   "model":model, 
                                        "pretrainNo": pretrainNo, 
                                        "pretrainDesc":pretrainDesc, 
                                        "noise":noise, 
                                        "temperature":temperature, 
                                        "decay":decay}
                            error, df = Optimize(param=param, args=args)
                            param['error'] = error
                            param['df'] = df
                            params.append(param)
                            pbar.update(1)
    pbar.close()
    errors = [param['error'] for param in params]
    bestIndex = np.argmin(errors)
    best = params[bestIndex]
    df = best['df']
    df.to_pickle("./Simulations/HIBL.pkl")

    df_means = df.groupby(['Description', 'Environment'], as_index=False)['Risky'].mean()
    guess = [
        df_means[(df_means['Environment'] == "Immediate") & (df_means['Description'] == 'No Description')]['Risky'].item(),
        df_means[(df_means['Environment'] == "Immediate") & (df_means['Description'] == 'Description')]['Risky'].item(),
        df_means[(df_means['Environment'] == "Clustered") & (df_means['Description'] == 'No Description')]['Risky'].item(),
        df_means[(df_means['Environment'] == "Clustered") & (df_means['Description'] == 'Description')]['Risky'].item(),
        df_means[(df_means['Environment'] == "Delayed") & (df_means['Description'] == 'No Description')]['Risky'].item(),
        df_means[(df_means['Environment'] == "Delayed") & (df_means['Description'] == 'Description')]['Risky'].item()
    ]
    print("Best guess: ", guess)
    
    palette = sns.color_palette(palette='tab10')
    palette = [palette[1], palette[0]]

    for param in params:
        param['df'] = 0

    print("Best parameters: ", best)
    sns.barplot(data=df, x="Environment", y="Risky", hue="Description", hue_order=["No Description", "Description"], palette=palette)
    plt.show()
            