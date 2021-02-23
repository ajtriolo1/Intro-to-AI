import math
import random
import numpy as np
from typing import Tuple
from heapq import heappush, heappop
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import time
from FireMaze import *

class Coord:
    def __init__(self, m: {np.ndarray, list}, x: int, y: int):
        """initialize the object
        :param m: a squared matrix. m[i][j] = 0 -> available, 1 -> occupied, -1 -> visited, 2 -> shortest path
        :param x: x-coord of the cell
        :param y: y-coord of the cell
        """
        self._m = m
        self._x = x
        self._y = y
        self._prev = None  # for tracing the path

    def is_available(self):
        """check whether the cell is available in the game
        :return: boolean
        """
        return self._m[self._x][self._y] == 0

    def get_neighbor_coords(self):
        x, y = (self._x, self._y)
        coords = [(x - 1, y), (x, y - 1), (x + 1, y), (x, y + 1)]
        dim = len(self._m)
        li = [(a, b) for (a, b) in coords if 0 <= a < dim and 0 <= b < dim and self._m[a][b] == 0]
        return li

    def close(self):
        """mark Coord as restricted.
        :return: void
        """
        # self._steps += 1  # increment _steps already taken from start to current location.
        self._m[self._x][self._y] = -1  # mark as visited

    def get_neighbors(self):
        """get the neighbor cells for the given cell
        :return: a list of available neighbor cells
        """
        return [Coord(self._m, x, y) for x, y in self.get_neighbor_coords()]

    def get_coords(self):
        """getter
        :return: coordinates of the cell
        """
        return self._x, self._y

    def set_prev_to(self, other: 'Coord'):
        """setter
        :param other: the previous cell
        :return: None
        """
        self._prev = other

    def get_path(self):
        """get the complete path from start to goal
        :return: a list of cells that form the path
        """
        cell = self
        path_nodes = []
        while cell:
            self._m[cell._x][cell._y] = 2  # mark the shortest path in matrix
            path_nodes = [(cell._x, cell._y)] + path_nodes  # insert to the front of the list
            cell = cell._prev
        return path_nodes

    def __str__(self):  # override
        return f'({self._x}, {self._y})'


class EuclideanCoord(Coord):
    def __init__(self, m: {np.ndarray, list}, x: int, y: int):
        """ inherit Coord
        :param m: matrix
        :param x: x-coord
        :param y: y-coord
        """
        super().__init__(m, x, y)
        self._steps = 0  # g(max_dim), informed search: f(max_dim) = g(max_dim) + h(max_dim)

    def close(self):  # override
        """mark Coord as restricted.
        :return: void
        """
        self._steps += 1  # increment _steps already taken
        super(EuclideanCoord, self).close()

    def get_neighbors(self):  # override
        """get the neighbor cells for the given cell
        :return: a list of available neighbor cells
        """
        return [EuclideanCoord(self._m, x, y) for x, y in super(EuclideanCoord, self).get_neighbor_coords()]

    def copy_steps_from(self, other: 'EuclideanCoord'):
        """setter
        :param other: the cell to copy from
        :return: None
        """
        self._steps = other._steps

    def __euclidean_distance(self):
        """heuristic
        :return: euclidean distance from the cell to the goal
        """
        return math.sqrt((len(self._m) - self._x - 1) ** 2 + (len(self._m) - self._y - 1) ** 2)

    def __lt__(self, other: 'EuclideanCoord'):
        """less than function, for peer comparison
        :param other: an EuclideanCoord object
        :return: whether this cell has better heuristic value over the other cell
        """
        return self._steps + self.__euclidean_distance() < other._steps + other.__euclidean_distance()

class SimCoord(Coord):
    def __init__(self, m: {np.ndarray, list}, x: int, y: int, fire_prob:list):
        """ inherit Coord
        :param m: matrix
        :param x: x-coord
        :param y: y-coord
        """
        super().__init__(m, x, y)
        self._steps = 0  # g(max_dim), informed search: f(max_dim) = g(max_dim) + h(max_dim)
        self._prob = fire_prob

    def close(self):  # override
        """mark Coord as restricted.
        :return: void
        """
        self._steps += 1  # increment _steps already taken
        super(SimCoord, self).close()

    def get_neighbors(self):  # override
        """get the neighbor cells for the given cell
        :return: a list of available neighbor cells
        """
        return [SimCoord(self._m, x, y, self._prob) for x, y in super(SimCoord, self).get_neighbor_coords()]

    def copy_steps_from(self, other: 'EuclideanCoord'):
        """setter
        :param other: the cell to copy from
        :return: None
        """
        self._steps = other._steps

    def __euclidean_distance(self):
        """heuristic
        :return: euclidean distance from the cell to the goal
        """
        return math.sqrt((len(self._m) - self._x - 1) ** 2 + (len(self._m) - self._y - 1) ** 2)

    def __lt__(self, other: 'SimCoord'):
        """less than function, for peer comparison
        :param other: a SimCoord object
        :return: whether this cell has better heuristic value over the other cell
        """
        return ((1/self._prob[self._x][self._y]) * (self._steps + self.__euclidean_distance())) < ((1/other._prob[other._x][other._y]) * (other._steps + other.__euclidean_distance()))

class MazeGame:
    def __init__(self, dimension: int, prob: float, fire_spread=0.0, m=np.array([])):
        """initialize the object
        :param dimension: dimension of the matrix that represents the maze
        :param prob: density of blocked cells in the maze
        :param fire_spread: fire spread rate
        :param m: maze to use rather than generating one
        """
        self._dim = dimension
        self._p = prob  # block density
        if len(m) != 0:
            self._m = m.copy()
        else:
            self._m = self.generate_maze()  # original matrix
        self._path = []  # list of Coords that form the path
        self._matrix = None  # result matrix
        self._fire_loc = [] # list of spaces that are on fire
        self._q = fire_spread # fire spread rate
        self._fire_prob = None # average number of steps for fire to spread to each point in simulations
        self._nodes_explored = None # number of nodes explored by an algorithm

    def generate_maze(self):
        """generate the matrix for the maze game
        :return: an numpy 2d array with 0s(available cells) and 1s(blocked cells)
        """
        mat = [[1 if random.uniform(0, 1) <= self._p else 0 for _ in range(self._dim)] for _ in range(self._dim)]
        mat[0][0] = 0 # starting node not blocked
        mat[self._dim - 1][self._dim - 1] = 0 # ending node not blocked
        return np.array(mat)
        

    def __is_reachable(self, start: Tuple[int, int], goal: Tuple[int, int], is_dfs: bool):
        """check whether goal(goal) is reachable from s(start) in the maze
        :param start: int tuple contains starting coordinates
        :param goal: int tuple contains goal coordinates
        :param is_dfs: True/False for dfs/bfs
        :return whether start and goal are reachable from each other
        """
        self._nodes_explored = 0
        self._matrix = self._m.copy()
        m = self._matrix
        if m[start[0]][start[1]] == 1 or m[goal[0]][goal[1]] == 1:
            return False  # not reachable because the start/target c is occupied.
        fringe = [Coord(m, start[0], start[1])]  # stack/queue
        while len(fringe) != 0:
            cell = fringe.pop(-1) if is_dfs else fringe.pop(0)  # dfs/bfs
            if not cell.is_available():
                continue
            cell.close()  # add restriction
            self._nodes_explored += 1
            if cell.get_coords() == goal:
                self._path = cell.get_path()
                return True
            else:  # add valid neighbor cells to fringe
                neighbors = cell.get_neighbors()
                for ne in neighbors:
                    ne.set_prev_to(cell)
                    fringe.append(ne)
        return False

    def dfs(self, start: Tuple[int, int], goal: Tuple[int, int]):
        return self.__is_reachable(start, goal, True)

    def bfs(self, start: Tuple[int, int], goal: Tuple[int, int]):
        return self.__is_reachable(start, goal, False)

    def a_star(self, start: Tuple[int, int], goal: Tuple[int, int]):
        """
        A* search: heuristic function = steps already taken + Euclidean distance to the goal
        :param start: int tuple contains starting coordinates
        :param goal: int tuple contains goal coordinates
        :return whether start and goal are reachable from each other
        """
        self._nodes_explored = 0
        self._matrix = self._m.copy()
        m = self._matrix
        if m[start[0]][start[1]] == 1 or m[goal[0]][goal[1]] == 1:
            return False  # not reachable because the start/target cell is occupied.
        heap = []  # priority heap/queue
        start_cell = EuclideanCoord(m, start[0], start[1])
        heappush(heap, start_cell)
        while len(heap) != 0:
            cell = heappop(heap)
            if not cell.is_available():
                continue
            cell.close()  # add restriction
            self._nodes_explored += 1
            if cell.get_coords() == goal:
                self._path = cell.get_path()
                return True
            else:  # add valid neighbor cells to fringe
                neighbors = cell.get_neighbors()
                for ne in neighbors:
                    ne.set_prev_to(cell)
                    ne.copy_steps_from(cell)
                    heappush(heap, ne)
        return False

    def get_path(self):
        """getter
        :return: a list of cells that form the shortest path
        """
        return self._path
    
    def get_nodes(self):
        """getter
        :return: number of nodes explored
        """
        return self._nodes_explored

    def get_original_matrix(self):
        """getter
        :return: original matrix
        """
        return self._m.copy()

    def get_result_matrix(self):
        """getter
        :return: None or final matrix
        """
        return self._matrix.copy()
    
    def set_fire(self):
        """setter: sets the initial fire location in a maze
        :return: the intial location of the fire
        """
        free_spaces = np.argwhere(self._m == 0) # list of unblocked cells in maze
        free_spaces = free_spaces[1:-1] # start and end nodes not considered
        index = random.randrange(0, len(free_spaces))
        free_x = free_spaces[index][0] # set random x value for initial fire
        free_y = free_spaces[index][1] # set random y value for initial fire
        if self.dfs((0,0), (free_x, free_y)) == False: # makes sure start node is reachable from this location
            return False
        self._m[free_x][free_y] = 3 # sets node in maze to fire value
        self._fire_loc.append((free_x, free_y))
        return (free_x, free_y)
       
    def start_fire_maze(self):
        """
        Sets the initial fire location and checks to make sure the goal is still reachable from the start
        """
        if(self.set_fire() == False):
            return False
        success = self.bfs((0,0), (self._dim-1, self._dim-1))
        return success

    def get_transformed_coords(self, coord: Tuple[int, int]):
        """
        Used to get neighbors of fire nodes
        prarm coord : the coordinates of the neighbors we want to look at
        returns: list of coordinates of neighboring nodes

        """
        x, y = coord
        transformed = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [(a, b) for (a, b) in transformed if 0 <= a < self._dim and 0 <= b < self._dim and
                self._m[a][b] != 1]

    def spread_fire(self):
        """
        Spreads fire in maze based on q probability
        """
        coords_on_fire = self._fire_loc
        fire_coords_with_available_neighbors = [] 
        for coord in coords_on_fire:
            transformed_coords = self.get_transformed_coords(coord)  # get adjacent coords
            available_coords = [(a, b) for (a, b) in transformed_coords if self._m[a][b] == 0]  # filter coords
            # if no potential firing neighbor, this coord won't proceed to the next round
            if len(available_coords) != 0:
                fire_coords_with_available_neighbors.append(coord)
            for curr_coord in available_coords:
                coords_to_check = self.get_transformed_coords(curr_coord)
                # count the numbers of firing neighbors
                k = sum(1 for curr_coord in coords_to_check if
                        self._m[curr_coord[0]][curr_coord[1]] == 3)
                curr_fire_prob = 1 - (1 - self._q) ** k  # probability of the current neighbor to be on fire.
                if random.uniform(0, 1) <= curr_fire_prob:
                    x, y = curr_coord
                    # the initial firing cell is marked as 1, so the following fire cells are marked by n in
                    # the nth round.
                    self._m[x][y] = 3
                    fire_coords_with_available_neighbors.append(curr_coord)
        self._fire_loc = fire_coords_with_available_neighbors  # replace the firing list for the next round
            
    def strat_1(self):
        """
        Strategy 1 of trying to survive the while moving through a maze
        """
        for point in self._path[1:]: # iterate through initial path discovered by bfs run when fire game is started
            if(point[0] == self._dim-1 and point[1] == self._dim-1):
                return True # agent reached goal
            self.spread_fire() # spread fire after taking a step
            for fire in self._fire_loc:
                if point[0] == fire[0] and point[1] == fire[1]: # check if agent is on fire
                    return False
                
    def strat_2(self):
        """
        Strategy 2 of trying to survive the while moving through a maze
        """
        if self._q == 0.0: # fire doesn't spread
            return True
        while len(self._path) > 1: # iterate through path found for each bfs run
            point = None
            point = self._path[1] # skip node we came from in path
            if(point[0] == self._dim-1 and point[1] == self._dim-1):
                return True
            self.spread_fire() # spread fire for this time step
            for fire in self._fire_loc:
               if point[0] == fire[0] and point[1] == fire[1]:
                   return False # agent on fire
            success = self.bfs((point[0],point[1]), (self._dim-1, self._dim-1)) # recalculate path
            if success == False:
                return False # no path available
    
    def simulation_assisted_search(self, start: Tuple[int, int], goal: Tuple[int, int]):
        """
        search: heuristic function = 1/number of steps before on fire in sims * (steps already taken + Euclidean distance to the goal)
        :param start: int tuple contains starting coordinates
        :param goal: int tuple contains goal coordinates
        :return whether start and goal are reachable from each other
        """
        self._matrix = self._m.copy()
        m = self._matrix
        if m[start[0]][start[1]] == 1 or m[goal[0]][goal[1]] == 1:
            return False  # not reachable because the start/target cell is occupied.
        heap = []  # priority heap
        start_cell = SimCoord(m, start[0], start[1], self._fire_prob)
        heappush(heap, start_cell)
        while len(heap) != 0:
            cell = heappop(heap)
            if not cell.is_available():
                continue
            cell.close()  # add restriction
            if cell.get_coords() == goal:
                self._path = None
                self._path = cell.get_path()
                return True
            else:  # add valid neighbor cells to fringe
                neighbors = cell.get_neighbors()
                for ne in neighbors:
                    ne.set_prev_to(cell)
                    ne.copy_steps_from(cell)
                    heappush(heap, ne)
        return False        
    
    def strat_3(self):
        """
        Strategy 3 of trying to survive the while moving through a maze using the new SAS algorithm
        """
        if self._q == 0.0:
            return True
        fire_maze = FireMaze(self._dim, self._p, self._q, self._m, self._fire_loc[0]) # new FireMaze module instance
        self._fire_prob = fire_maze.simulate_fire(10) # run simulations of fire spread
        self.simulation_assisted_search((0,0), (self._dim-1, self._dim-1)) # run search using probabilites found during simulation
        for point in self._path[1:]:
            if(point[0] == self._dim-1 and point[1] == self._dim-1):
                return True
            self.spread_fire()
            if (self._dim-1, self._dim-1) in self._fire_loc:
                return False # goal is on fire
            for fire in self._fire_loc:
                if point[0] == fire[0] and point[1] == fire[1]:
                    return False # agent is on fire
        
def test_strat_1(n):
    """
    Runs reuqired tests for strategy 1
    :param n: size of mazes
    """
    average_success_per_q = [] # average number of successful mazes out of 10 for each q
    q=0.0
    while q/10 <= 1.0:
        print("Running mazes for q=", q/10)
        success_per_maze = [] # Number of successful mazes for 10 different initial fire locations
        i=1
        while i <= 10:
            success = 0
            game = MazeGame(n, 0.3, q/10)
            maze = game.get_original_matrix()
            if game.dfs((0,0), (n-1,n-1)) == False:
                continue # discard mazes where no path from start to finish
            j=1
            while j<= 10:
                fire_game = MazeGame(n, 0.3, q/10, maze) # start with new initial fire location
                if fire_game.start_fire_maze() == False:
                    continue # discard mazes where no path from fire to start
                if fire_game.strat_1() == True: # Run strategy 1
                    success+=1
                    j+=1
                else:
                    j+=1
            success_per_maze.append(success)
            i+=1
        maze_success = 0
        for success in success_per_maze:
            maze_success+=success
        average_success_per_q.append(maze_success/10) # Get average number of successful mazes out of 10
        q+=1
        
    """Plot statistics"""    
    q_vals = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
    avg_success = np.array(average_success_per_q)
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(q_vals, avg_success)
    ax.set_title('Average Successes vs Fire Spread Rate (Strategy 1)', fontdict={'fontsize': 18, 'fontweight': 'medium'})
    plt.yticks(np.arange(0, avg_success.max()+1, 1))
    plt.xlabel('Fire Spread Rate (q)', fontsize=12)
    plt.ylabel('Average Successes', fontsize=12)
    plt.show()
    
def test_strat_2(n):
    """
    Runs reuqired tests for strategy 2
    :param n: size of mazes
    """
    average_success_per_q = [] # average number of successful mazes out of 10 for each q
    q=0.0
    while q/10 <= 1.0:
        print(q/10)
        success_per_maze = [] # Number of successful mazes for 10 different initial fire locations
        i=1
        while i <= 10:
            success = 0
            game = MazeGame(n, 0.3, q/10)
            maze = game.get_original_matrix()
            if game.dfs((0,0), (n-1,n-1)) == False:
                continue # discard mazes where no path from start to finish
            j=1
            while j<= 10:
                fire_game = MazeGame(n, 0.3, q/10, maze) # start with new initial fire location
                if fire_game.start_fire_maze() == False:
                    continue # discard mazes where no path from fire to start
                if fire_game.strat_2() == True: # Run strategy 2
                    success+=1
                    j+=1
                else:
                    j+=1
            success_per_maze.append(success)
            i+=1
        maze_success = 0
        for success in success_per_maze:
            maze_success+=success
        average_success_per_q.append(maze_success/10) # Get average number of successful mazes out of 10
        q+=1
    
    """Plot statistics""" 
    q_vals = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
    avg_success = np.array(average_success_per_q)
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(q_vals, avg_success)
    ax.set_title('Average Successes vs Fire Spread Rate (Strategy 2)', fontdict={'fontsize': 18, 'fontweight': 'medium'})
    plt.yticks(np.arange(0, avg_success.max()+1, 1))
    plt.xlabel('Fire Spread Rate (q)', fontsize=12)
    plt.ylabel('Average Successes', fontsize=12)
    plt.show()

def test_strat_3(n):
    average_success_per_q = [] # average number of successful mazes out of 10 for each q
    q=0.0
    while q/10 <= 1.0:
        print(q/10)
        success_per_maze = [] # Number of successful mazes for 10 different initial fire locations
        i=1
        while i <= 10:
            success = 0
            game = MazeGame(n, 0.3, q/10)
            maze = game.get_original_matrix()
            if game.dfs((0,0), (n-1,n-1)) == False:
                continue # discard mazes where no path from start to finish
            j=1
            while j<= 10:
                fire_game = MazeGame(n, 0.3, q/10, maze) # start with new initial fire location
                if fire_game.start_fire_maze() == False:
                    continue # discard mazes where no path from fire to start
                if fire_game.strat_3() == True: # Run strategy 3
                    success+=1
                    j+=1
                else:
                    j+=1
            success_per_maze.append(success)
            i+=1
        maze_success = 0
        for success in success_per_maze:
            maze_success+=success
        average_success_per_q.append(maze_success/10) # Get average number of successful mazes out of 10
        q+=1
        
    """Plot statistics""" 
    q_vals = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
    avg_success = np.array(average_success_per_q)
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(q_vals, avg_success)
    ax.set_title('Average Successes vs Fire Spread Rate (Strategy 3)', fontdict={'fontsize': 18, 'fontweight': 'medium'})
    plt.yticks(np.arange(0, avg_success.max()+1, 1))
    plt.xlabel('Fire Spread Rate (q)', fontsize=12)
    plt.ylabel('Average Successes', fontsize=12)
    plt.show()

def test_dfs(n, trials):
    """
    Tests success probability of dfs at different blocking density values
    :param n: size of mazes
    :param trials: number of mazes to try
    """
    success_prob_per_p = [] # Success probability at each p value
    p = 0.0
    while p/10 <= 1.0: # for each p value
        print(p/10)
        i=1
        success = 0
        while i<=trials:
            game = MazeGame(n, p/10) # Try new maze
            if game.dfs((0,0), (n-1, n-1)) == False: # Failure
                i+=1
                continue
            else: #Success
                i+=1
                success+=1
        success_prob_per_p.append(success/trials) # Calculate success rate for given p
        p+=1
        
    """ Plot Graph"""
    p_vals = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
    success_prob = np.array(success_prob_per_p)
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(p_vals, success_prob)
    ax.set_title('Success Probability vs Obstacle Density', fontdict={'fontsize': 18, 'fontweight': 'medium'})
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('Obstacle Density (p)', fontsize=12)
    plt.ylabel('Success Probability', fontsize=12)
    plt.show()
    
def test_bfs_a_star(n, num_trials):
    """
    Tests average difference in nodes explored between BFS and A* at different p values
    :param n: size of mazes
    :param trials: number of mazes to test
    """
    average_node_dif = [] # Average difference at each p value
    p = 0.0
    while p/10 <= 1.0: # for each p value
        print(p/10)
        i=1
        node_dif = 0
        while i<=num_trials:
            game = MazeGame(n, p/10) # Try new maze
            game.bfs((0,0), (n-1,n-1))
            bfs_nodes = game.get_nodes() # Nodes explored by BFS
            game.a_star((0,0), (n-1,n-1))
            a_star_nodes = game.get_nodes() # Nodes explored by A*
            node_dif += (bfs_nodes - a_star_nodes) # Difference in nodes explored
            i+=1
        average_node_dif.append(node_dif/num_trials) # Average difference for given p
        p+=1
    
    """ Plot Graph"""
    p_vals = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
    avg_node_dif = np.array(average_node_dif)
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(p_vals, avg_node_dif)
    ax.set_title('(BFS Nodes - A* Nodes) vs Obstacle Density', fontdict={'fontsize': 18, 'fontweight': 'medium'})
    plt.yticks(np.arange(0, avg_node_dif.max()+n, n))
    plt.xlabel('Obstacle Density (p)', fontsize=12)
    plt.ylabel('BFS Nodes - A* Nodes', fontsize=12)
    plt.show()

def test_time_bfs():
    """
    Tests maximum size for which BFS can solve a maze in under a minute
    """
    time_consumed = 0
    size = 2500 # Initial size of maze
    while time_consumed <= 60: # Check that mazes are being solved in under a minute
        i=1
        while i <= 5: # Run 5 trials
            game = MazeGame(size, 0.3) # New maze
            time_start = time.time()
            if game.bfs((0,0), (size-1, size-1)) == False: # Make sure to only use successfu mazes
                continue
            time_end = time.time()
            time_consumed += time_end-time_start # Add time it took to run bfs
            i+=1
        time_consumed = time_consumed/5 # Calculate average amount of time
        print(size, ":", time_consumed)
        size += 100 # Increase size of maze
    print("Largest BFS size:", size-200, "x", size-200)
    
def test_time_dfs():
    """
    Tests maximum size for which DFS can solve a maze in under a minute
    """
    time_consumed = 0
    size = 7000 # Initial size of maze
    while time_consumed <= 60: # Check that mazes are being solved in under a minute
        i=1
        while i <= 5: # Run 5 trials
            game = MazeGame(size, 0.3) # New maze
            time_start = time.time()
            if game.dfs((0,0), (size-1, size-1)) == False: # Make sure to only use successfu mazes
                continue
            time_end = time.time()
            print(time_end-time_start)
            time_consumed += time_end-time_start # Add time it took to run dfs
            i+=1
        time_consumed = time_consumed/5 # Calculate average amount of time
        print(size, ":", time_consumed)
        size += 100 # Increase size of maze
    print("Largest DFS size:", size-200, "x", size-200)

def test_time_a_star():
    """
    Tests maximum size for which BFS can solve a maze in under a minute
    """
    time_consumed = 0
    size = 1000 # Initial size of maze
    while time_consumed <= 60: # Check that mazes are being solved in under a minute
        i=1
        while i <= 5: # Run 5 trials
            game = MazeGame(size, 0.3)
            time_start = time.time()
            if game.a_star((0,0), (size-1, size-1)) == False: # Make sure to only use successfu mazes
                continue
            time_end = time.time()
            time_consumed += time_end-time_start # Add time it took to run dfs
            i+=1
        time_consumed = time_consumed/5 # Calculate average amount of time
        print(size, ":", time_consumed)
        size += 50 # Increase size of maze
    print("Largest A* size:", size-100, "x", size-100)    
    
# Test
if __name__ == '__main__':
    n = 250
    
    """Run Tests"""
    test_dfs(n, 25)
    test_bfs_a_star(n, 25)
    test_time_bfs()
    test_time_dfs()
    test_time_a_star()
    test_strat_1(75)
    test_strat_2(75)
    test_strat_3(75)
            