import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import argparse
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
import numpy as np
import random 

from Models import IBLAgent, FRLAgent

def Train(args):
    cols = ["Name", "Hierachical", "Timestep", "Reward", "Correct", "Fit Params"]
    df = pd.DataFrame([], columns=cols)
    


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    parser.add_argument("--dataPath", type=str, default="/Data/study1.csv", help="Path to human decisions.")
    

    args = parser.parse_args()

    print(Train(args))
            