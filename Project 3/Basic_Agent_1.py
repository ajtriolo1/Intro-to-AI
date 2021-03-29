from Environment import Environment
import numpy as np
import random
import time
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class Agent:
    def __init__(self, env: Environment, agent_type: int):
        self.env = env
        self.map = env.get_map()
        self.dim = self.map.shape[0]
        self.p_f = {1:1/10, 2:3/10, 3:7/10, 4:9/10}
        self.beliefs = np.full(self.map.shape, 1/(self.dim*self.dim))
        self.init_prob = 1/(self.dim*self.dim)
        self.observed = {}
        self.type = agent_type
        
    def update_belief(self, cell: (int, int), p_obs):
        cell_type = self.map[cell[0]][cell[1]]
        if self.type == 1:
            self.beliefs[cell[0]][cell[1]] = (self.p_obs_given_cell(cell) * self.init_prob)/p_obs
        if self.type == 2:
            self.beliefs[cell[0]][cell[1]] = (1-self.p_f[cell_type])*((self.p_obs_given_cell(cell) * self.init_prob)/p_obs)
        
    def p_obs_given_cell(self, cell: (int, int)):
        if cell in self.observed.keys():
            cell_x, cell_y = cell
            cell_type = self.map[cell_x][cell_y]
            return self.p_f[cell_type]**self.observed[cell]
        else:
            return 1

    
    def get_p_obs(self, cell: (int, int), p_obs):
        cell_type = self.map[cell[0]][cell[1]]
        if p_obs == 0:
            return self.p_f[cell_type]*self.init_prob + ((self.dim**2-1)*self.init_prob)
        else:
            return p_obs-(self.p_f[cell_type]**(self.observed[cell]-1)-self.p_f[cell_type]**self.observed[cell])*self.init_prob
    
    def run_game(self):
        num_steps = 1
        num_search = 1
        p_obs = 0
        map_x = random.randint(0, self.dim-1)
        map_y = random.randint(0, self.dim-1)
        cell = (map_x, map_y)
        if not self.env.is_target(cell):
            self.observed[cell] = 1
            p_obs = self.get_p_obs(cell, p_obs)
            for x in range(self.dim):
                for y in range(self.dim):
                    self.update_belief((x, y), p_obs)
        else:
            return num_steps + num_search
        print(self.beliefs)
        while True:
            largest = np.where(self.beliefs == np.amax(self.beliefs))
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
                if cell in self.observed.keys():
                    self.observed[cell] += 1
                else:
                    self.observed[cell] = 1
                p_obs = self.get_p_obs(cell, p_obs)
                for x in range(self.dim):
                    for y in range(self.dim):
                        self.update_belief((x, y), p_obs)
            else:
                return num_steps + num_search
            print(self.beliefs)
        
if __name__ == '__main__':
    agent_1_stats = 0
    agent_2_stats = 0
    for mazes in range(10):
        print("Maze", mazes+1)
        env = Environment(50)
        agent_1_trial = 0
        agent_2_trial = 0
        for trials in range(10):
            env.set_target()
            agent_1 = Agent(env, 1)
            agent_2 = Agent(env, 2)
            print("\tTrial", trials+1, "for agent 1")
            agent_1_trial += agent_1.run_game()
            print("\tTrial", trials+1, "for agent 2")
            agent_2_trial += agent_2.run_game()
        agent_1_stats += agent_1_trial/10
        agent_2_stats += agent_2_trial/10
    agent_1_stats /= 10
    agent_2_stats /= 10
    
    x = ['Basic Agent 1', 'Basic Agent 2']
    x_pos = [i for i, _ in enumerate(x)]
    plt.bar(x_pos, [agent_1_stats, agent_2_stats], color=['blue', 'red'])
    plt.xlabel('Agent Type')
    plt.ylabel('Average Score (Steps + Searches)')
    plt.title('Average score for each agent')
    plt.xticks(x_pos, x)
    plt.show()

    """
    env = Environment(2)
    env.set_target()
    env.print_map()
    env.print_target()
    agent_1 = Agent(env, 1)
    agent_2 = Agent(env, 2)
    print(agent_1.run_game())
    print(agent_2.run_game())
    """