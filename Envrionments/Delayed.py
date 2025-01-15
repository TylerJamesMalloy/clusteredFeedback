from . import Environment 

class Delayed(Environment):
    def __init__(self, args):
        self.name = "Delayed"
        super(Environment, self).__init__(args)

    def state(self):
        print(self.name) 
    
    def reward(self,action):
        return 
    
