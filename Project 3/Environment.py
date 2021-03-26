import numpy as np
import random

class Environment:
    def __init__(self, d:int):
        self.dim = d
        self.map = self.build_map()
        self.target = self.set_target()
        self.f_n = {1:0.1, 2:0.3, 3:0.7, 4:0.9}
        
    def build_map(self):
        game_map = np.zeros([self.dim, self.dim], dtype=int)
        for x in range(self.dim):
            for y in range(self.dim):
                p = random.uniform(0,1)
                if (p < 0.25):
                    game_map[x][y] = 1
                    continue
                elif (p < 0.5):
                    game_map[x][y] = 2
                    continue
                elif (p < .75):
                    game_map[x][y] = 3
                    continue
                else:
                    game_map[x][y] = 4
                    continue
        return game_map
                
    def set_target(self):
        x = random.randint(0, self.dim-1)
        y = random.randint(0, self.dim-1)
        
        return (x, y)
    
    def is_target(self, cell: (int, int)):
        target_x, target_y = self.target
        target_type = self.map[target_x][target_y]
        if cell == self.target:
            f_n = self.f_n[target_type]
            p = random.uniform(0, 1)
            if p < f_n:
                return False
            else:
                return True
        else:
            return False
            
         
    def print_map(self):
        print(self.map)
        
    def get_map(self):
        return self.map
        
    def get_target(self):
        return self.target
    
    def print_target(self):
        print(self.target)
        