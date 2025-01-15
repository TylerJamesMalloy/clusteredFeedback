from . import Bandit 

class UCBAgent(Bandit):
    def choose(self,options):
        print(options)
    
    def respond(self,reward):
        print (reward)
    

