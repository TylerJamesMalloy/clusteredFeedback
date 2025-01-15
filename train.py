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
    for timestep in range(args.timesteps):
        reward, risky = envir.step(agent.choose(state=envir.state()))
        df = pd.concat(df, pd.DataFrame([[envir.name, args.descr, agent.name, args.ident, timestep, reward, risky]], columns=cols), ignore_index=True)
    return df
    


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    parser.add_argument("--dataPath", type=str, default="/HumanBehavior/study1.csv", help="Path to human decisions.")
    parser.add_argument("--trace", type=bool, default=False, help="Whether or not to trace the human decisions in dataPath")
    parser.add_argument("--model", type=str, default="HIBLAgent", 
                        choices=['HIBLAgent', 'HRLAgent', 'HTSAgent', 'HUCBAgent', 'IBLAgent', 'RLAgent', 'TSAgent', 'UCBAgent'],
                        help="Name of the agent to use")
    #parser.add_argument("--envir", type=str, default="Immediate", choices=['Immediate', 'Clustered', 'Delayed'], help="Name of the agent to use")

    args = parser.parse_args()
    
    cols = ["Environment", "Description", "Name", "Agent ID", "Timestep", "Reward", "Risky"]
    df = pd.DataFrame([], columns=cols)

    for ident in tqdm.tqdm(range(1000)):
        for envir in ['Immediate', 'Clustered', 'Delayed']:
            for descr in [True, False]:
                args.envir = envir 
                args.descr = descr
                args.ident = ident
                df = pd.concat(df, Train(args), ignore_index=True)
    if(args.trace):
        df.to_pickle("./ModelTracing/Results.pkl")
    else:
        df.to_pickle("./Simulations/Results.pkl")
            