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
    args.model = param['model']
    args.pretrainNo = param['pretrainNo']
    args.pretrainDesc = param['pretrainDesc']
    args.noise = param['noise']
    args.temperature = param['temperature']
    args.decay = param['decay']
    
    envirs = ['Immediate', 'Delayed', 'Clustered']
    #envirs = ['Immediate']
    descrs = ["Description", "No Description"]

    cols = ["Environment", "Description", "Name", "Agent ID", "Timestep", "Reward", "Observed Reward", "Risky"]
    df = pd.DataFrame([], columns=cols)
    for ident in range(args.agents):
        for envir in envirs:
            for descr in descrs:
                args.envir = envir 
                args.descr = descr
                args.ident = model + " " + str(ident)
                args.model = model
                args.param = param
                df = pd.concat([df, Train(args)], ignore_index=True)

    guess = df.groupby(['Environment', 'Description'], as_index=False)['Risky'].mean()['Risky'].to_numpy()
    print(model, "  ", guess)

    return df

"""
IBL Best parameters:  {'model': 'IBLAgent', 'pretrainNo': 0, 'pretrainDesc': 100, 'noise': 0.2, 'temperature': 0.5, 'decay': 0.3, 'error': np.float64(0.023640000000000015), 'df': 0}
HIBL Best parameters:  {'model': 'HIBLAgent', 'pretrainNo': 0, 'pretrainDesc': 25, 'noise': 0.2, 'temperature': 0.5, 'decay': 0.1, 'error': np.float64(0.022615), 'df': 0}
"""
def Train(args):
    cols = ["Environment", "Description", "Name", "Agent ID", "Timestep", "Reward", "Observed Reward", "Risky"]
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
    parser.add_argument("--agents", type=int, default=50, help="Num Agents")
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
    
    cols = ["Environment", "Description", "Name", "Agent ID", "Timestep", "Reward", "Observed Reward", "Risky"]
    df = pd.DataFrame([], columns=cols)

    #tqdm.tqdm()
    param = {   "model":"HIBLAgent", 
                "pretrainNo": 0, 
                "pretrainDesc":50, 
                "noise":0.2, 
                "temperature":0.5, 
                "decay":0.1}

    df = Optimize(param=param, args=args)
    df.to_pickle("./Results/HIBL.pkl")

    param = {   "model":"IBLAgent", 
                "pretrainNo": 0, 
                "pretrainDesc":100, 
                "noise":0.2, 
                "temperature":0.5, 
                "decay":0.3}

    df = Optimize(param=param, args=args)
    df.to_pickle("./Results/IBL.pkl")

    
            