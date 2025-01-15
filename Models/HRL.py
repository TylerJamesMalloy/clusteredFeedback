from . import Bandit 

class HRLAgent(Bandit):
    def choose(self,options):
        print(options)
    
    def respond(self,reward):
        print (reward)
    

