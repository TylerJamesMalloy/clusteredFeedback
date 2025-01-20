import numpy as np 

class Delayed():
    def __init__(self, args):
        self.name = "Delayed"
        self.rewardSum = 0
        self.args = args 
        self.ts = 0
    
    def options(self):
        self.ts += 1
        return ["A", "B"]
    
    def reset(self):
        self.rewardSum = 0
        self.ts = 0

    def reward(self,action):
        risky = False if action == "A" else True
        if(self.ts % self.args.window == 0):
            true_reward =  4 if action == "A" else int(np.random.choice([0,10], 1, p=[0.5, 0.5])[0]) 
            self.rewardSum += true_reward
            return (self.rewardSum, true_reward, risky)
        else:
            true_reward =  4 if action == "A" else int(np.random.choice([0,10], 1, p=[0.5, 0.5])[0]) 
            self.rewardSum += true_reward
            return (None, true_reward, risky) 
    
