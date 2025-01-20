from pyibl import Agent, DelayedResponse
import numpy as np 

"""
Best guess:  [0.307, 0.673, 0.546, 0.694, 0.569, 0.718]
Best parameters:  {'model': 'IBLAgent', 'pretrainNo': 0, 'pretrainDesc': 25, 'noise': 0.2, 'temperature': 0.5, 'decay': 0.1, 'error': np.float64(0.022615), 'df': 0}
"""
class IBLAgent():
    def __init__(self, args):
        self.args = args
        self.name = "IBL"
        self.a = Agent(name='Agent', default_utility=4.5, noise=self.args.noise, temperature=self.args.temperature, decay=self.args.decay)
        self.delayedResponses = []
        

    def pretrain(self):
        if(self.args.descr == "Description"): 
            for _ in range(self.args.pretrainDesc):
                self.a.populate(["A"], self.args.sure)
                #self.a.populate(["B"], 5)
                self.a.populate(["B"], int(np.random.choice([0,10], 1, p=[0.5, 0.5])[0]))
        if(self.args.descr == "No Description"): 
            for _ in range(self.args.pretrainNo):
                self.a.populate(["A"], self.args.sure)
                #self.a.populate(["B"], 5)
                self.a.populate(["B"], int(np.random.choice([0,10], 1, p=[0.5, 0.5])[0]))

    def choose(self,options):
        return self.a.choose(options)
    
    def modelTrace(self, observedReward, modelRisky, humanRisky, humanReward):
        _, options, decisions, values = self.a._pending_decision
        humanChoice = 0 if humanRisky else 1
        self.a._pending_decision = (humanChoice, options, decisions, values)

        if(self.args.envir == "Immediate"):
            self.a.respond(humanReward)
        elif(self.args.envir == "Delayed"):
            if(observedReward == None):
                self.delayedResponses.append(self.a.respond())
            else:
                self.delayedResponses.append(self.a.respond())
                for delayedResponse in self.delayedResponses:
                    delayedReward = (observedReward/self.args.window) #+ np.random.normal(0,0.1)
                    delayedResponse.update(delayedReward)
                self.delayedResponses = []
        elif(self.args.envir == "Clustered"):
            if(observedReward == None):
                self.delayedResponses.append(self.a.respond())
            else:
                self.delayedResponses.append(self.a.respond())
                for delayedResponse, outcome in zip(self.delayedResponses, observedReward):
                    delayedResponse.update(outcome)
                self.delayedResponses = []
        else:
            print("Environment not recognized in IBL Agent")
            assert(False)
    
    def respond(self,reward):
        if(self.args.envir == "Immediate"):
            self.a.respond(reward)
        elif(self.args.envir == "Delayed"):
            if(reward == None):
                self.delayedResponses.append(self.a.respond())
            else:
                self.delayedResponses.append(self.a.respond())
                for delayedResponse in self.delayedResponses:
                    delayedReward = (reward/self.args.window) #+ np.random.normal(0,0.1)
                    delayedResponse.update(delayedReward)
                self.delayedResponses = []
        elif(self.args.envir == "Clustered"):
            if(reward == None):
                self.delayedResponses.append(self.a.respond())
            else:
                self.delayedResponses.append(self.a.respond())
                for delayedResponse, outcome in zip(self.delayedResponses, reward):
                    delayedResponse.update(outcome)
                self.delayedResponses = []
        else:
            print("Environment not recognized in IBL Agent")
            assert(False)
    

