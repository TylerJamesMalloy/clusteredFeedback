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
def Train(args):
    cols = ["Environment", "Description", "Name", "Agent ID", "Timestep", "Reward", "Risky"]
    df = pd.DataFrame([], columns=cols)
    agent = getattr(sys.modules[__name__], args.model)(args)
    envir = getattr(sys.modules[__name__], args.envir)(args)
    agent.pretrain()
    for timestep in range(args.timesteps):
        reward, risky = envir.reward(agent.choose(options=envir.options()))
        agent.respond(reward)
        df = pd.concat([df, pd.DataFrame([[envir.name, args.descr, agent.name, args.ident, timestep, reward, risky]], columns=cols)], ignore_index=True)
    return df

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    parser.add_argument("--dataPath", type=str, default="/HumanBehavior/study1.csv", help="Path to human decisions.")
    parser.add_argument("--trace", type=bool, default=False, help="Whether or not to trace the human decisions in dataPath")
    parser.add_argument("--agents", type=int, default=50, help="Num Agents")
    parser.add_argument("--window", type=int, default=10, help="Feedback window size")
    parser.add_argument("--risky", type=list, default=[[0,10],[0.5,0.5]], help="Probabilities for risky option (B).")
    parser.add_argument("--sure", type=int, default=4, help="Value of the sure option (A).")
    parser.add_argument("--pretrainNo", type=int, default=0, help="Pretraining steps for description environment.")
    parser.add_argument("--pretrainDesc", type=int, default=200, help="Pretraining steps for description environment.")
    parser.add_argument("--noise", type=float, default=0.25, help="IBL noise parameter")
    parser.add_argument("--decay", type=float, default=0.5, help="IBL noise parameter")
    parser.add_argument("--temperature", type=float, default=0.35, help="IBL noise parameter")
    parser.add_argument("--timesteps", type=int, default=100, help="Num timesteps")
    args = parser.parse_args()

    #DEFAULT_NOISE = 0.25
    #DEFAULT_DECAY = 0.5
    #DEFAULT_TEMPERATURE = np.sqrt(2) * 0.25
    
    cols = ["Environment", "Description", "Name", "Agent ID", "Timestep", "Reward", "Risky"]
    df = pd.DataFrame([], columns=cols)
    models = ["HIBLAgent"]
    envirs = ['Immediate', 'Delayed', 'Clustered']
    descrs = ["Description", "No Description"]

    for ident in tqdm.tqdm(range(args.agents)):
        for model in models:
            for envir in envirs:
                for descr in descrs:
                    args.envir = envir 
                    args.descr = descr
                    args.ident = ident
                    args.model = model
                    df = pd.concat([df, Train(args)], ignore_index=True)
    
    print(df)
    palette = sns.color_palette(palette='tab10')
    palette = [palette[1], palette[0]]
    fig, axes = plt.subplots(nrows=2, ncols=2, sharey=True)
    sns.barplot(df, x="Environment", y="Risky", hue="Description", palette=palette, hue_order=["No Description", "Description"], ax=axes[0][0])

    axes[0][0].set_title("IBL")
    #axes[0][0].set_ylim(0,1)
    axes[0][0].set_ylabel("Choice of the Risky Option")
    plt.show()
    if(args.trace):
        df.to_pickle("./ModelTracing/Results.pkl")
    else:
        df.to_pickle("./Simulations/Results.pkl")
            