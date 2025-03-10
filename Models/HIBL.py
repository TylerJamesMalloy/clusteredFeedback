from pyibl import Agent, DelayedResponse
import numpy as np 
"""
Best guess:  [0.358, 0.691, 0.573, 0.625, 0.64, 0.771]
true =       [0.40,  0.75,   0.60, 0.75,  0.60, 0.75]
Best parameters:  {'model': 'IBLAgent', 'pretrainNo': 0, 'pretrainDesc': 100, 'noise': 0.2, 'temperature': 0.5, 'decay': 0.3, 'error': np.float64(0.023640000000000015), 'df': 0}
"""

class HIBLAgent():
    def __init__(self, args):
        self.args = args
        self.a = Agent(name='Agent', default_utility=4.5, noise=self.args.noise, temperature=self.args.temperature, decay=self.args.decay)
        
        self.a1 = Agent(name='A', default_utility=4, decay=0, noise=0, temperature=0.1)
        self.a2 = Agent(name='B', default_utility=10, decay=1, noise=1, temperature=10)
        self.chosen = None 
        
        self.agents = [self.a1, self.a2]
        self.detailHistory = []
        self.name = "HIBL"
        self.ts = 0
        self.delayedResponses = []

    def pretrain(self):
        if(self.args.descr == "Description"): 
            for _ in range(self.args.pretrainDesc):
                self.a.populate([1], self.args.sure)
                self.a.populate([2], int(np.random.choice([0,10], 1, p=[0.5, 0.5])[0]))
                self.a1.populate(["A"], self.args.sure)
                self.a2.populate(["B"], int(np.random.choice([0,10], 1, p=[0.5, 0.5])[0]))
        if(self.args.descr == "No Description"): 
            for _ in range(self.args.pretrainNo):
                self.a.populate([1], self.args.sure)
                self.a.populate([2], int(np.random.choice([0,10], 1, p=[0.5, 0.5])[0]))
                self.a1.populate(["A"], self.args.sure)
                self.a2.populate(["B"], int(np.random.choice([0,10], 1, p=[0.5, 0.5])[0]))  

    def choose(self, options):
        self.ts += 1
        if(self.chosen is None):
            choice, details = self.a.choose([1, 2], details=True)

        self.detailHistory.append(details)
        if(choice == 1):
            choice, details = self.agents[0].choose(options, details=True) 
            self.chosen = self.agents[0]
            self.detailHistory.append(details)
        else:
            choice, details = self.agents[1].choose(options, details=True) 
            self.chosen = self.agents[1]
            self.detailHistory.append(details)
        self.choice = choice 
        return choice
    
    def modelTrace(self, observedReward, modelRisky, humanRisky, humanReward):
        _, options, decisions, values = self.a._pending_decision
        humanChoice = 0 if humanRisky else 1
        self.a._pending_decision = (humanChoice, options, decisions, values)

        if(self.args.envir == "Immediate"):
            self.a.respond(humanReward)
            for agent in self.agents:
                if(self.chosen._pending_decision is not None):
                    agent._pending_decision = tuple((self.chosen._pending_decision[1].index(self.choice), self.chosen._pending_decision[1], self.chosen._pending_decision[2], self.chosen._pending_decision[3]))
                    agent.respond(humanReward)
        elif(self.args.envir == "Delayed"):
            self.a.respond(humanReward)
            if(self.chosen == self.agents[1]):
                self.agents[0]._pending_decision = tuple((self.chosen._pending_decision[1].index(self.choice), self.chosen._pending_decision[1], self.chosen._pending_decision[2], self.chosen._pending_decision[3]))
            else:
                self.agents[1]._pending_decision = tuple((self.chosen._pending_decision[1].index(self.choice), self.chosen._pending_decision[1], self.chosen._pending_decision[2], self.chosen._pending_decision[3]))
                
            if(observedReward == None):
                self.delayedResponses.append([self.agents[0].respond(), self.agents[1].respond()])
            else:
                self.delayedResponses.append([self.agents[0].respond(), self.agents[1].respond()])
                for delayedResponse in self.delayedResponses:
                    for response in delayedResponse:
                        delayedReward = (observedReward/self.args.window) #+ np.random.normal(0,0.1)
                        response.update(delayedReward)
                self.delayedResponses = []
        elif(self.args.envir == "Clustered"):
            if(self.chosen == self.agents[1]):
                self.agents[0]._pending_decision = tuple((self.chosen._pending_decision[1].index(self.choice), self.chosen._pending_decision[1], self.chosen._pending_decision[2], self.chosen._pending_decision[3]))
            else:
                self.agents[1]._pending_decision = tuple((self.chosen._pending_decision[1].index(self.choice), self.chosen._pending_decision[1], self.chosen._pending_decision[2], self.chosen._pending_decision[3]))
            self.a.respond(np.sum(observedReward))
            if(observedReward == None):
                self.delayedResponses.append([self.agents[0].respond(), self.agents[1].respond()])
            else:
                self.delayedResponses.append([self.agents[0].respond(), self.agents[1].respond()])
                for delayedResponse in self.delayedResponses:
                    for r, response in zip(observedReward, delayedResponse):
                        delayedReward = (r) #+ np.random.normal(0,0.1)
                        response.update(delayedReward)
                self.delayedResponses = []
        else:
            print("Environment not recognized in HIBL model")
            assert(False)

    def respond(self, reward, humanChoice=None):
        if(self.args.envir == "Immediate"):
            self.a.respond(reward)
            for agent in self.agents:
                if(self.chosen._pending_decision is not None):
                    agent._pending_decision = tuple((self.chosen._pending_decision[1].index(self.choice), self.chosen._pending_decision[1], self.chosen._pending_decision[2], self.chosen._pending_decision[3]))
                    agent.respond(reward)
        elif(self.args.envir == "Delayed"):
            self.a.respond(reward)
            if(self.chosen == self.agents[1]):
                self.agents[0]._pending_decision = tuple((self.chosen._pending_decision[1].index(self.choice), self.chosen._pending_decision[1], self.chosen._pending_decision[2], self.chosen._pending_decision[3]))
            else:
                self.agents[1]._pending_decision = tuple((self.chosen._pending_decision[1].index(self.choice), self.chosen._pending_decision[1], self.chosen._pending_decision[2], self.chosen._pending_decision[3]))
                
            if(reward == None):
                self.delayedResponses.append([self.agents[0].respond(), self.agents[1].respond()])
            else:
                self.delayedResponses.append([self.agents[0].respond(), self.agents[1].respond()])
                for delayedResponse in self.delayedResponses:
                    for response in delayedResponse:
                        delayedReward = (reward/self.args.window) #+ np.random.normal(0,0.1)
                        response.update(delayedReward)
                self.delayedResponses = []
        elif(self.args.envir == "Clustered"):
            self.a.respond(np.sum(reward))
            if(self.chosen == self.agents[1]):
                self.agents[0]._pending_decision = tuple((self.chosen._pending_decision[1].index(self.choice), self.chosen._pending_decision[1], self.chosen._pending_decision[2], self.chosen._pending_decision[3]))
            else:
                self.agents[1]._pending_decision = tuple((self.chosen._pending_decision[1].index(self.choice), self.chosen._pending_decision[1], self.chosen._pending_decision[2], self.chosen._pending_decision[3]))
                
            if(reward == None):
                self.delayedResponses.append([self.agents[0].respond(), self.agents[1].respond()])
            else:
                self.delayedResponses.append([self.agents[0].respond(), self.agents[1].respond()])
                for delayedResponse in self.delayedResponses:
                    for r, response in zip(reward, delayedResponse):
                        delayedReward = (r) #+ np.random.normal(0,0.1)
                        response.update(delayedReward)
                self.delayedResponses = []
        else:
            print("Environment not recognized in HIBL model")
            assert(False)
        return reward