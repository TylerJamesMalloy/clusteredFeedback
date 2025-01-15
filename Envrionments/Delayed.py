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
            return (self.rewardSum, risky)
        else:
            self.rewardSum += 4 if action == "A" else np.random.choice([0,10], 1, p=[0.5, 0.5])[0] 
            return (None, risky) 
    
