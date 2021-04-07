from Environment import Environment
import numpy as np
import random
import time
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class Agent_1:
    def __init__(self, env: Environment):
        self.env = env
        self.map = env.get_map()
        self.dim = self.map.shape[0]
        self.p_f = {1:1/10, 2:3/10, 3:7/10, 4:9/10}
        self.beliefs = np.full(self.map.shape, 1/(self.dim*self.dim))
        self.init_prob = 1/(self.dim*self.dim)
        self.observed = (0,0)
    
    def update_belief(self, cell: (int, int), b_t):
        cell_type = self.map[self.observed[0]][self.observed[1]]
        if cell == self.observed:
            self.beliefs[cell[0]][cell[1]] = self.beliefs[cell[0]][cell[1]]*self.p_f[cell_type]/(b_t*self.p_f[cell_type] + (1-b_t))
        else:
            self.beliefs[cell[0]][cell[1]] = self.beliefs[cell[0]][cell[1]]/(b_t*self.p_f[cell_type] + (1-b_t))
    
    def run_game(self):
        num_steps = 1
        num_search = 1
        map_x = random.randint(0, self.dim-1)
        map_y = random.randint(0, self.dim-1)
        cell = (map_x, map_y)
        if not self.env.is_target(cell):
            self.observed = cell
            b_t = self.beliefs[self.observed[0]][self.observed[1]]
            for x in range(self.dim):
                for y in range(self.dim):
                    self.update_belief((x, y), b_t)
        else:
            return num_steps, num_search
        while True:
            largest = np.where(np.isclose(self.beliefs,np.max(self.beliefs)))
            largest_coords = list(zip(largest[0], largest[1]))
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
        self.env = env
        self.map = env.get_map()
        self.dim = self.map.shape[0]
        self.p_f = {1:1/10, 2:3/10, 3:7/10, 4:9/10}
        self.beliefs = np.full(self.map.shape, 1/(self.dim*self.dim))
        self.init_prob = 1/(self.dim*self.dim)
        self.observed = (0,0)
        self.p_found = np.empty(self.map.shape)
    
    def update_belief(self, cell: (int, int), b_t):
        cell_type = self.map[self.observed[0]][self.observed[1]]
        if cell == self.observed:
            self.beliefs[cell[0]][cell[1]] = self.beliefs[cell[0]][cell[1]]*self.p_f[cell_type]/(b_t*self.p_f[cell_type] + (1-b_t))
        else:
            self.beliefs[cell[0]][cell[1]] = self.beliefs[cell[0]][cell[1]]/(b_t*self.p_f[cell_type] + (1-b_t))
            
    def run_game(self):
        num_steps = 1
        num_search = 1
        map_x = random.randint(0, self.dim-1)
        map_y = random.randint(0, self.dim-1)
        cell = (map_x, map_y)
        if not self.env.is_target(cell):
            self.observed = cell
            b_t = self.beliefs[self.observed[0]][self.observed[1]]
            for x in range(self.dim):
                for y in range(self.dim):
                    self.update_belief((x, y), b_t)
                    self.p_found[x][y] = (1-self.p_f[self.map[x][y]])* self.beliefs[x][y]
        else:
            return num_steps, num_search
        while True:
            largest = np.where(np.isclose(self.p_found, np.max(self.p_found)))
            largest_coords = list(zip(largest[0], largest[1]))
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
            
class Agent_3:
    def __init__(self, env: Environment):
        self.env = env
        self.map = env.get_map()
        self.dim = self.map.shape[0]
        self.p_f = {1:1/10, 2:3/10, 3:7/10, 4:9/10}
        self.beliefs = np.full(self.map.shape, 1/(self.dim*self.dim))
        self.init_prob = 1/(self.dim*self.dim)
        self.observed = (0,0)
        self.p_found = np.empty(self.map.shape)
    
    def update_belief(self, cell: (int, int), b_t):
        cell_type = self.map[self.observed[0]][self.observed[1]]
        if cell == self.observed:
            self.beliefs[cell[0]][cell[1]] = self.beliefs[cell[0]][cell[1]]*self.p_f[cell_type]**cell_type/(b_t*self.p_f[cell_type]**cell_type + (1-b_t))
        else:
            self.beliefs[cell[0]][cell[1]] = self.beliefs[cell[0]][cell[1]]/(b_t*self.p_f[cell_type]**cell_type + (1-b_t))
            
    def run_game(self):
        num_steps = 1
        num_search = 0
        map_x = random.randint(0, self.dim-1)
        map_y = random.randint(0, self.dim-1)
        cell = (map_x, map_y)
        cell_type = self.map[cell[0]][cell[1]]
        self.observed = cell
        b_t = self.beliefs[self.observed[0]][self.observed[1]]
        for i in range(cell_type):
            num_search += 1
            if self.env.is_target(cell):
                return num_steps, num_search
        for x in range(self.dim):
            for y in range(self.dim):
                self.update_belief((x, y), b_t)
                self.p_found[x][y] = (1-self.p_f[self.map[x][y]]) * self.beliefs[x][y]
        while True:
            largest = np.where(np.isclose(self.p_found, np.max(self.p_found)))
            largest_coords = list(zip(largest[0], largest[1]))
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
            cell_type = self.map[cell[0]][cell[1]]
            self.observed = cell
            b_t = self.beliefs[self.observed[0]][self.observed[1]]
            for i in range(cell_type):
                num_search += 1
                if self.env.is_target(cell):
                    return num_steps, num_search
            for x in range(self.dim):
                for y in range(self.dim):
                    self.update_belief((x, y), b_t)
                    self.p_found[x][y] = (1-self.p_f[self.map[x][y]]) * self.beliefs[x][y]
    
        
if __name__ == '__main__':
    """
    agent_1_steps = 0
    agent_1_searches = 0
    agent_2_steps = 0
    agent_2_searches = 0
    types = {1:0, 2:0, 3:0, 4:0}
    for maze in range(10):
        env = Environment(50)
        print("Maze", maze+1)
        for i in range(10):
            env.set_target()
            types[env.get_target_type()] += 1
            env.print_target()
            agent_1 = Agent_1(env)
            agent_2 = Agent_2(env)
            print('Trial', i+1, 'for agent 1')
            agent_1_result = agent_1.run_game()
            print('\t Done in', agent_1_result[0], 'steps and', agent_1_result[1], 'searches')    
            agent_1_steps += agent_1_result[0]
            agent_1_searches += agent_1_result[1]
            print('Trial', i+1, 'for agent 2')
            agent_2_result = agent_2.run_game()
            print('\t Done in', agent_2_result[0], 'steps and', agent_2_result[1], 'searches')    
            agent_2_steps += agent_2_result[0]
            agent_2_searches += agent_2_result[1]
    agent_1_score = (agent_1_steps + agent_1_searches)/100
    agent_1_steps /= 100
    agent_1_searches /= 100
    agent_2_score = (agent_2_steps + agent_2_searches)/100
    agent_2_steps /= 100
    agent_2_searches /= 100
    print('Agent 1 had', agent_1_steps, 'steps and', agent_1_searches, 'searches on average for a score of', agent_1_score)
    print('Agent 2 had', agent_2_steps, 'steps and', agent_2_searches, 'searches on average for a score of', agent_2_score)
    print(types)
    """
    agent_1_steps = 0
    agent_1_searches = 0
    agent_2_steps = 0
    agent_2_searches = 0
    agent_3_steps = 0
    agent_3_searches = 0
    types = {1:0, 2:0, 3:0, 4:0}
    env = Environment(50)
    for i in range(10):
        env.set_target()
        types[env.get_target_type()] += 1
        env.print_target()
        agent_1 = Agent_1(env)
        agent_2 = Agent_2(env)
        agent_3 = Agent_3(env)
        print('Trial', i+1, 'for agent 1')
        agent_1_result = agent_1.run_game()
        print('\t Done in', agent_1_result[0], 'steps and', agent_1_result[1], 'searches')    
        agent_1_steps += agent_1_result[0]
        agent_1_searches += agent_1_result[1]
        print('Trial', i+1, 'for agent 2')
        agent_2_result = agent_2.run_game()
        print('\t Done in', agent_2_result[0], 'steps and', agent_2_result[1], 'searches')    
        agent_2_steps += agent_2_result[0]
        agent_2_searches += agent_2_result[1]
        print('Trial', i+1, 'for agent 3')
        agent_3_result = agent_3.run_game()
        print('\t Done in', agent_3_result[0], 'steps and', agent_3_result[1], 'searches')    
        agent_3_steps += agent_3_result[0]
        agent_3_searches += agent_3_result[1]
    agent_1_score = (agent_1_steps + agent_1_searches)/10
    agent_1_steps /= 10
    agent_1_searches /= 10
    agent_2_score = (agent_2_steps + agent_2_searches)/10
    agent_2_steps /= 10
    agent_2_searches /= 10
    agent_3_score = (agent_3_steps + agent_3_searches)/10
    agent_3_steps /= 10
    agent_3_searches /= 10
    print('Agent 1 had', agent_1_steps, 'steps and', agent_1_searches, 'searches on average for a score of', agent_1_score)
    print('Agent 2 had', agent_2_steps, 'steps and', agent_2_searches, 'searches on average for a score of', agent_2_score)
    print('Agent 3 had', agent_3_steps, 'steps and', agent_3_searches, 'searches on average for a score of', agent_3_score)
    print(types)


    
    
    