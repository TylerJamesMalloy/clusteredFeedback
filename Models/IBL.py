from pyibl import Agent, DelayedResponse
from . import Bandit 

class IBLAgent(Bandit):
    def choose(self,options):
        print(options)
    
    def respond(self,reward):
        print (reward)
    

