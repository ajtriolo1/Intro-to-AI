from Environment import Environment
import numpy as np
import random
import time
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class Agent_1:
    def __init__(self, env: Environment):
        self.env = env #Environment to query
        self.map = env.get_map() # Map of types of cells
        self.dim = self.map.shape[0]
        self.p_f = {1:1/10, 2:3/10, 3:7/10, 4:9/10} # False negative probabilties
        self.beliefs = np.full(self.map.shape, 1/(self.dim*self.dim)) # Belief matrix
        self.init_prob = 1/(self.dim*self.dim)
        self.observed = (0,0) # Last observed cell
    
    def update_belief(self, cell: (int, int), b_t):
        cell_type = self.map[self.observed[0]][self.observed[1]]
        if cell == self.observed: # Cell was the one searched
            self.beliefs[cell[0]][cell[1]] = self.beliefs[cell[0]][cell[1]]*self.p_f[cell_type]/(b_t*self.p_f[cell_type] + (1-b_t))
        else:
            self.beliefs[cell[0]][cell[1]] = self.beliefs[cell[0]][cell[1]]/(b_t*self.p_f[cell_type] + (1-b_t))
    
    def run_game(self):
        num_steps = 1
        num_search = 1
        map_x = random.randint(0, self.dim-1)
        map_y = random.randint(0, self.dim-1)
        cell = (map_x, map_y) # Random cell to start
        if not self.env.is_target(cell): # Check if target is found at cell
            self.observed = cell
            b_t = self.beliefs[self.observed[0]][self.observed[1]] # Observed cell's belief
            for x in range(self.dim):
                for y in range(self.dim):
                    self.update_belief((x, y), b_t) # Update belief of each cell based on last observation
        else:
            return num_steps, num_search
        while True:
            largest = np.where(np.isclose(self.beliefs,np.max(self.beliefs)))
            largest_coords = list(zip(largest[0], largest[1]))
            # Find the cell with highest probability and lowest distance if needed
            if len(largest_coords) > 1:
                distances = list()            
                for coord in largest_coords:
                    distances.append(abs(coord[0]-cell[0]) + abs(coord[1]-cell[1]))
                distances = np.array(distances)
                shortest = np.where(distances == distances.min())
                shortest = list(shortest[0])
                distances = distances.tolist()
                if len(shortest) > 1:
                    index = random.randint(0, len(shortest)-1)
                    num_steps += distances[shortest[index]]
                    cell = largest_coords[shortest[index]]
                else: 
                    num_steps += distances[shortest[0]]
                    cell = largest_coords[shortest[0]]
            else:
                num_steps += abs(largest_coords[0][0]-cell[0]) + abs(largest_coords[0][1]-cell[1])
                cell = largest_coords[0]
            num_search += 1
            if not self.env.is_target(cell):
                self.observed = cell
                b_t = self.beliefs[self.observed[0]][self.observed[1]]
                for x in range(self.dim):
                    for y in range(self.dim):
                        self.update_belief((x, y), b_t)
            else:
                return num_steps, num_search
            
class Agent_2:
    def __init__(self, env: Environment):
        self.env = env #Environment to query
        self.map = env.get_map() # Map of types of cells
        self.dim = self.map.shape[0]
        self.p_f = {1:1/10, 2:3/10, 3:7/10, 4:9/10} # False negative probabilties
        self.beliefs = np.full(self.map.shape, 1/(self.dim*self.dim)) # Belief matrix
        self.init_prob = 1/(self.dim*self.dim)
        self.observed = (0,0) # Last observed cell
        self.p_found = np.empty(self.map.shape) # Probabilities of finding target in cells
    
    def update_belief(self, cell: (int, int), b_t):
        cell_type = self.map[self.observed[0]][self.observed[1]] # Cell type of last searched cell
        if cell == self.observed: # Cell was the one searched
            self.beliefs[cell[0]][cell[1]] = self.beliefs[cell[0]][cell[1]]*self.p_f[cell_type]/(b_t*self.p_f[cell_type] + (1-b_t))
        else:
            self.beliefs[cell[0]][cell[1]] = self.beliefs[cell[0]][cell[1]]/(b_t*self.p_f[cell_type] + (1-b_t))
            
    def run_game(self):
        num_steps = 1
        num_search = 1
        map_x = random.randint(0, self.dim-1)
        map_y = random.randint(0, self.dim-1)
        cell = (map_x, map_y) # Random cell to start
        if not self.env.is_target(cell): # Check if target is found at cell
            self.observed = cell
            b_t = self.beliefs[self.observed[0]][self.observed[1]] # Observed cell's belief
            for x in range(self.dim):
                for y in range(self.dim):
                    self.update_belief((x, y), b_t) # Update belief of each cell based on last observation
                    self.p_found[x][y] = (1-self.p_f[self.map[x][y]])* self.beliefs[x][y] # Update probability of finding target in each cell
        else:
            return num_steps, num_search
        while True:
            largest = np.where(np.isclose(self.p_found, np.max(self.p_found)))
            largest_coords = list(zip(largest[0], largest[1]))
            # Get cell with highest probability of finding target and shortest distance if needed
            if len(largest_coords) > 1:
                distances = list()            
                for coord in largest_coords:
                    distances.append(abs(coord[0]-cell[0]) + abs(coord[1]-cell[1]))
                distances = np.array(distances)
                shortest = np.where(distances == distances.min())
                shortest = list(shortest[0])
                distances = distances.tolist()
                if len(shortest) > 1:
                    index = random.randint(0, len(shortest)-1)
                    num_steps += distances[shortest[index]]
                    cell = largest_coords[shortest[index]]
                else: 
                    num_steps += distances[shortest[0]]
                    cell = largest_coords[shortest[0]]
            else:
                num_steps += abs(largest_coords[0][0]-cell[0]) + abs(largest_coords[0][1]-cell[1])
                cell = largest_coords[0]
            num_search += 1
            if not self.env.is_target(cell):
                self.observed = cell
                b_t = self.beliefs[self.observed[0]][self.observed[1]]
                for x in range(self.dim):
                    for y in range(self.dim):
                        self.update_belief((x, y), b_t)
                        self.p_found[x][y] = (1-self.p_f[self.map[x][y]]) * self.beliefs[x][y]
            else:
                return num_steps, num_search
          
