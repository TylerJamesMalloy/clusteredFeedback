import numpy as np 

class Immediate():
    def __init__(self, args):
        self.name = "Immediate"
        self.type = "Immediate"
        self.args = args 
        self.ts = 0

    def reset(self):
        self.ts = 0

    def options(self):
        self.ts += 1
        return ["A", "B"]
    
    def reward(self,action):
        true_reward = int(np.random.choice([0,10], 1, p=[0.5, 0.5])[0])
        return (4, 4, False) if action == "A" else (true_reward, true_reward, True)
         
    
