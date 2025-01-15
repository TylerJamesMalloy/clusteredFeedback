from pyibl import Agent, DelayedResponse

class HIBLAgent():
    def __init__(self, args):
        self.args = args
        self.a = Agent(name='Agent', default_utility=4.5)
        a1 = Agent(name='A', default_utility=0, decay=0, noise=0, temperature=10)
        a2 = Agent(name='B', default_utility=10, decay=1, noise=1, temperature=0.1)
        self.agents = [a1, a2]
        self.detailHistory = []
        self.name = "HIBL"

    def choose(self, options):
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
    
    def respond(self, reward, humanChoice=None):
        if(self.args.envir == "Immediate"):
            self.a.respond(reward)
            for agent in self.agents:
                if(self.chosen._pending_decision is not None):
                    agent._pending_decision = tuple((self.chosen._pending_decision[1].index(self.choice), self.chosen._pending_decision[1], self.chosen._pending_decision[2], self.chosen._pending_decision[3]))
                    agent.respond(reward)
        elif(self.args.envir == "Delayed"):
            for agent in self.agents:
                agent._pending_decision = tuple((self.chosen._pending_decision[1].index(self.choice), self.chosen._pending_decision[1], self.chosen._pending_decision[2], self.chosen._pending_decision[3]))
                if(self.ts % self.args.window == 0):
                    for agent in self.agents:
                        agent.respond(reward)
                else:
                    for agent in self.agents:
                        agent.respond()
                if(self.args.trace):
                    if(self.choice == humanChoice):
                        self.a.respond(1)
                    else:
                        self.a.respond(0)
                else:
                    self.a.respond(reward)
        elif(self.args.envir == "Clustered"):
            print("Not initialized") 
        else:
            print("Environment not recognized in HIBL model")
            assert(False)
        return reward
        
    

