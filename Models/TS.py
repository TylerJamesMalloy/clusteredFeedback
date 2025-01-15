from . import Bandit 

class TSAgent(Bandit):
    def choose(self,options):
        print(options)
    
    def respond(self,reward):
        print (reward)
    

