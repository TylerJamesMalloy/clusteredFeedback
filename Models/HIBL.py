from pyibl import Agent, DelayedResponse
from . import Bandit 

class HIBLAgent(Bandit):
    def choose(self,options):
        print(options)
    
    def respond(self,reward):
        print (reward)
    

