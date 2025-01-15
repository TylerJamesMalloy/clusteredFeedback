from . import Bandit 

class RLAgent(Bandit):
    def choose(self,options):
        print(options)
    
    def respond(self,reward):
        print (reward)
    

