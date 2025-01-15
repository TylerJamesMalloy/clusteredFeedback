from . import Environment 

class Immediate(Environment):
    def __init__(self, args):
        self.name = "Immediate"
        super(Environment, self).__init__(args)

    def state(self):
        print(self.name) 
    
    def reward(self,action):
        return 
    
