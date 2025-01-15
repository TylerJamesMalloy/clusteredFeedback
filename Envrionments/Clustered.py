from . import Environment 

class Clustered(Environment):
    def __init__(self, args):
        self.name = "Clustered"
        super(Environment, self).__init__(args)

    def state(self):
        print(self.name) 
    
    def reward(self,action):
        return 
    
