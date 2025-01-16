from pyibl import Agent, DelayedResponse
import numpy as np 

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
    

