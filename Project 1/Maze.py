import math
import random
import numpy as np
from typing import Tuple
from heapq import heappush, heappop
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


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
        return self._m[self._x][self._y] == 0 or self._m[self._x][self._y] == -2

    def get_neighbor_coords(self):
        x, y = (self._x, self._y)
        coords = [(x, y + 1), (x - 1, y), (x + 1, y), (x, y - 1)]
        dim = len(self._m)
        li = [(a, b) for (a, b) in coords if 0 <= a < dim and 0 <= b < dim and (self._m[a][b] == 0 or self._m[a][b] == -2)]
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


class MazeGame:
    def __init__(self, dimension: int, prob: float, fire_spread: float, m=np.array([])):
        """initialize the object
        :param dimension: dimension of the matrix that represents the maze
        :param prob: density of blocked cells in the maze
        """
        self._dim = dimension
        self._p = prob  # block density
        if len(m) != 0:
            self._m = m.copy()
        else:
            self._m = self.generate_maze()  # original matrix
        self._path = []  # list of Coords that form the path
        self._matrix = None  # result matrix
        self._fire_loc = []
        self._q = fire_spread

    def generate_maze(self):
        """generate the matrix for the maze game
        :return: an numpy 2d array with 0s(available cells) and 1s(blocked cells)
        """
        mat = [[1 if random.uniform(0, 1) <= self._p else 0 for _ in range(self._dim)] for _ in range(self._dim)]
        mat[0][0] = 0
        mat[self._dim - 1][self._dim - 1] = 0
        return np.array(mat)
        

    def __is_reachable(self, start: Tuple[int, int], goal: Tuple[int, int], is_dfs: bool):
        """check whether goal(goal) is reachable from s(start) in the maze
        :param start: int tuple contains starting coordinates
        :param goal: int tuple contains goal coordinates
        :param is_dfs: True/False for dfs/bfs
        :return whether start and goal are reachable from each other
        """
        
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
        A* search: heuristic function = Euclidean distance to the goal
        :param start: int tuple contains starting coordinates
        :param goal: int tuple contains goal coordinates
        :return whether start and goal are reachable from each other
        """
        self._matrix = self._m.copy()
        m = self._matrix
        if m[start[0]][start[1]] == 1 or m[goal[0]][goal[1]] == 1:
            return False  # not reachable because the start/target cell is occupied.
        heap = []  # priority heap
        start_cell = EuclideanCoord(m, start[0], start[1])
        heappush(heap, start_cell)
        while len(heap) != 0:
            cell = heappop(heap)
            if not cell.is_available():
                continue
            cell.close()  # add restriction
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
       free_spaces = np.argwhere(self._m == 0)
       free_spaces = free_spaces[1:-1]
       index = random.randrange(0, len(free_spaces))
       free_x = free_spaces[index][0]
       free_y = free_spaces[index][1]
       self._m[free_x][free_y] = 3
       self._fire_loc.append((free_x, free_y))
       return (free_x, free_y)
       
    def start_fire_maze(self):
        self.set_fire()
        success = self.bfs((0,0), (self._dim-1, self._dim-1))
        self.plot_fire()
        return success
        
    def spread_fire(self):
        free_spaces = np.argwhere(self._m != 1)
        for space in free_spaces:
            fire_count = 0
            x = space[0]
            y = space[1]
            free_cell = Coord(self._m, space[0], space[1])
            coords = [(x, y + 1), (x - 1, y), (x + 1, y), (x, y - 1)]
            dim = len(self._m)
            neighbor_coords = [(a, b) for (a, b) in coords if 0 <= a < dim and 0 <= b < dim and self._m[a][b] != 1]
            neighbors = [Coord(self._m, x, y) for x, y in neighbor_coords]
            for neighbor in neighbors:
                if self._m[neighbor.get_coords()[0]][neighbor.get_coords()[1]] == 3:
                    fire_count += 1
            if random.uniform(0, 1) <= (1-((1-self._q)**fire_count)):
                self._fire_loc.append((free_cell.get_coords()[0], free_cell.get_coords()[1]))
        for fire in self._fire_loc:
            self._m[fire[0]][fire[1]] = 3
        
    def strat_1(self):
        for point in self._path:
            self.plot_spread()
            for fire in self._fire_loc:
                if point[0] == fire[0] and point[1] == fire[1]:
                    return False
            if point[0] == 0 and point[1] == 0:
                self._m[point[0]][point[1]] = -2
                continue
            else:
                self._m[point[0]][point[1]] = -2
                self.spread_fire()
                if point[0] == fire[0] and point[1] == fire[1]:
                    return False
                
        return True
    
    def strat_2(self):
        self._m[0][0] = -2
        while len(self._path) != 1:
            point = self._path[1]
            print("Now at ", (point[0], point[1]))
            self.spread_fire()
            self.plot_spread()
            for fire in self._fire_loc:
               if point[0] == fire[0] and point[1] == fire[1]:
                   print("Step on fire at: ", (point[0], point[1]))
                   return False
            else:
                success = self.bfs((point[0],point[1]), (self._dim-1, self._dim-1))
                if success == False:
                    self.plot_fire()
                    print("No path")
                    return False
                self._m[point[0]][point[1]] = -2
                self.plot_fire()
                print(self._path)
                if point[0] == fire[0] and point[1] == fire[1]:
                    self.plot_spread()
                    return False
        self.plot_spread()
        return True                    
        
    def plot(self):
        """plot the matrix: yellow - visited cells, white - available cells, grey - blocked cells, green - shortest path
        :return: None
        """
        color_list = ['#ffff00', 'w', '#808080', 'g']  # [-1 yellow, 0 white, 1 grey, 2 green]
        colors = ListedColormap(color_list if len(self._path) > 0 else color_list[:-1])
        plt.matshow(self._matrix, interpolation='none', cmap=colors)
        plt.show()
        
    def plot_fire(self):
        color_list = ['#800080', '#ffff00', 'w', '#808080', 'g', '#FFA500']  # [-1 yellow, 0 white, 1 grey, 2 green, 3 orange]
        no_path_colors = ['#ffff00', 'w', '#808080', '#FFA500']
        colors = ListedColormap(color_list if len(self._path) > 0 else no_path_colors)
        plt.matshow(self._matrix, interpolation='none', cmap=colors)
        plt.show()
        
    def plot_spread(self):
        color_list = ['#800080', '#ffff00', 'w', '#808080', 'g', '#FFA500']  # [-1 yellow, 0 white, 1 grey, 2 green, 3 orange]
        no_path_colors = ['w', '#808080', '#FFA500']
        colors = ListedColormap(color_list if len(self._path) > 0 else no_path_colors)
        plt.matshow(self._m, interpolation='none', cmap=colors)
        plt.show()
        
# Test
if __name__ == '__main__':
    n = 10
    p = 0.1
    q = 0.1
    '''This will be what we use for testing strats
    for i in range (10): 
        game = MazeGame(n, p, q)
        m = game.get_original_matrix()
        for j in range (10):
            fire_game = MazeGame(n, p, q, m)
            fire_game.run_game()
    '''
    game = MazeGame(n, p, q)
    maze = game.get_original_matrix()
    game.start_fire_maze()
    game.strat_2()
    '''
    for i in range (5):
        print(i)
        fire_game = MazeGame(n, p, q, maze)
        fire_game.start_fire_maze()
        fire_game.strat_1()
    '''
    #print(game.a_star((0, 0), (n - 1, n - 1)))
    
