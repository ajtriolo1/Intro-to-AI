from Environment import Environment
import numpy as np
import random
import time

class Agent:
    def __init__(self, env: Environment):
        self.env = env
        self.map = env.get_map()
        self.dim = self.map.shape[0]
        self.p_f = {1:1/10, 2:3/10, 3:7/10, 4:9/10}
        self.beliefs = np.full(self.map.shape, 1/(self.dim*self.dim))
        self.init_prob = 1/(self.dim*self.dim)
        self.observed = {}
        
    def update_belief(self, cell: (int, int), p_obs):
        self.beliefs[cell[0]][cell[1]] = (self.p_obs_given_cell(cell) * self.init_prob)/p_obs
        
    def p_obs_given_cell(self, cell: (int, int)):
        for obs in self.observed.keys():
            cell_x, cell_y = cell
            cell_type = self.map[cell_x][cell_y]
            if obs == cell:
                return self.p_f[cell_type]**self.observed[obs]
        return 1    
    
    def get_p_obs(self):
        p_obs = 0
        for cell in self.observed.keys():
            p_obs += (self.init_prob * self.p_obs_given_cell(cell))
        if len(self.observed) < self.dim*self.dim:
            p_obs += self.init_prob * (self.dim*self.dim-len(self.observed))
        return p_obs
    
    def run_game(self):
        num_steps = 1
        num_search = 1
        map_x = random.randint(0, self.dim-1)
        map_y = random.randint(0, self.dim-1)
        cell = (map_x, map_y)
        if not self.env.is_target(cell):
            self.observed[cell] = 1
            p_obs = self.get_p_obs()
            for x in range(self.dim):
                for y in range(self.dim):
                    self.update_belief((x, y), p_obs)
        else:
            return num_steps, num_search
        while True:
            largest = np.where(self.beliefs == np.amax(self.beliefs))
            largest_indices = list(zip(largest[0], largest[1]))
            if len(largest_indices) > 1:
                distances = list()
                for indices in largest_indices:
                    distances.append(abs(indices[0]-cell[0]) + abs(indices[1]-cell[1]))
                shortest = [i for i, x in enumerate(distances) if x == min(distances)]
                if len(shortest) > 1:
                    index = random.randint(0, len(shortest)-1)
                    num_steps += distances[shortest[index]]
                    cell = largest_indices[shortest[index]]
                else: 
                    num_steps += distances[shortest[0]]
                    cell = largest_indices[shortest[0]]
            else:
                num_steps += abs(largest_indices[0][0]-cell[0]) + abs(largest_indices[0][1]-cell[1])
                cell = largest_indices[0]
            
            num_search += 1
            if not self.env.is_target(cell):
                if cell in self.observed.keys():
                    self.observed[cell] += 1
                else:
                    self.observed[cell] = 1
                p_obs = self.get_p_obs()
                for x in range(self.dim):
                    for y in range(self.dim):
                        self.update_belief((x, y), p_obs)
            else:
                return num_steps, num_search
        
        
if __name__ == '__main__':
    env = Environment(50)
    env.set_target()
    env.print_map()
    env.print_target()
    agent = Agent(env)
    time_start = time.time()
    print(agent.run_game())       
    time_end = time.time()
    print(time_end - time_start)
    
    