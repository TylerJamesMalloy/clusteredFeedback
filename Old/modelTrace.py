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
    hdf = param['df']
    envir = param['envir']
    descr = param['descr']
    
    args.envir = envir 
    args.descr = descr
    args.ident = hdf['id'].unique()[0]
    args.model = model
    args.param = param
    result = ModelTrace(args, hdf)
    error = 1 - (np.sum(result["Correct Prediction"]) / len(result["Correct Prediction"]))
    return error, result

"""
IBL Best parameters:  {'model': 'IBLAgent', 'pretrainNo': 0, 'pretrainDesc': 100, 'noise': 0.2, 'temperature': 0.5, 'decay': 0.3, 'error': np.float64(0.023640000000000015), 'df': 0}
HIBL Best parameters:  {'model': 'IBLAgent', 'pretrainNo': 0, 'pretrainDesc': 25, 'noise': 0.2, 'temperature': 0.5, 'decay': 0.1, 'error': np.float64(0.022615), 'df': 0}
"""
def ModelTrace(args, hdf):
    cols = ["Environment", "Description", "Name", "Agent ID", "Timestep", "True Reward", "Observed Reward", "Correct Prediction", "Model Risky", "Human Risky", "Human Reward"]
    df = pd.DataFrame([], columns=cols)
    agent = getattr(sys.modules[__name__], args.model)(args)
    envir = getattr(sys.modules[__name__], args.envir)(args)
    agent.pretrain()
    for timestep, (humanRisky, humanReward) in enumerate(zip(pdf['riskyOption'], pdf['payoff'])):
        observedReward, trueReward, modelRisky = envir.reward(agent.choose(options=envir.options()))
        correct = 1 if bool(humanRisky) == modelRisky else 0
        agent.modelTrace(observedReward, modelRisky, humanRisky, humanReward)
        df = pd.concat([df, pd.DataFrame([[envir.name, args.descr, agent.name, args.ident, timestep, trueReward, observedReward, correct, modelRisky, humanRisky, humanReward]], columns=cols)], ignore_index=True)
        
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
    
    cols = ["Environment", "Description", "Name", "Agent ID", "Timestep", "True Reward", "Observed Reward", "Correct Prediction", "Model Risky", "Human Risky"]
    df = pd.DataFrame([], columns=cols)

    study1 = pd.read_csv("./HumanBehavior/study1.csv")
    study2 = pd.read_csv("./HumanBehavior/study2.csv")
    study1['Environment'] = "Immediate"
    study2['Environment'] = "Immediate"
    study1.loc[study1['treatment'] == "Clustered Feedback", 'Environment'] = "Clustered"
    study2.loc[study2['treatment'] == "Clustered Feedback", 'Environment'] = "Delayed"
    params = []
    data = pd.concat([study1, study2])
    pretrainNos = [0, 50, 100]
    pretrainDescs = [100, 150, 200]
    noises = [0.1, 0.25,  0.5]
    temperatures = [0.1, 0.25,  0.5]
    decays = [0.1, 0.25,  0.5]
    models = ["HIBLAgent", "IBLAgent"] #,    

    pbar = tqdm.tqdm(total=len(data['id'].unique()) * len(pretrainNos) * len(pretrainDescs) * len(noises) * len(temperatures) * len(decays) * len(models))
    #pbar = tqdm.tqdm(total=10 * len(pretrainNos) * len(pretrainDescs) * len(noises) * len(temperatures) * len(decays) * len(models))
    bestDf = None 
    for id in data['id'].unique():
        pdf = data[data['id'] == id]
        descr = pdf['description'].unique()[0]
        for envir in pdf['Environment'].unique():
            if(envir == "Immediate"): continue 
            edf = pdf[pdf['Environment'] == envir]
            bestError = None
            bestDf = None 
            for pretrainNo in pretrainNos:
                for pretrainDesc in pretrainDescs:
                    for noise in noises:
                        for temperature in temperatures:
                            for decay in decays:
                                for model in models:
                                    param = {   "model":model, 
                                                "pretrainNo": pretrainNo, 
                                                "pretrainDesc":pretrainDesc, 
                                                "noise":noise, 
                                                "temperature":temperature, 
                                                "decay":decay,
                                                "df": edf}
                                    
                                    param['envir'] = envir
                                    param['descr'] = descr
                                    param['id'] = id
                                    param['model'] = model
                                    error, d = Optimize(param=param, args=args)
                                    pbar.update(1)
                                    if(bestError == None or error < bestError):
                                        bestError = error
                                        df['params'] = param
                                        bestDf = d
        if(bestDf is not None):
            bestDf.to_pickle("./Results/Participants/" + id + ".pkl")                               
            df = pd.concat([df, bestDf], ignore_index=True)

    #df.dropna(inplace=True)                                  
    pbar.close()
    df.to_pickle("Results/ModelTracing_Clustered.pkl")
    ibl = df[df['Name'] == 'IBL']
    print(ibl['Correct Prediction'].mean())
    hibl = df[df['Name'] == 'HIBL']
    print(hibl['Correct Prediction'].mean())


            