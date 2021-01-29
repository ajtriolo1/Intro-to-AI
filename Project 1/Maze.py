import random
from matplotlib import pyplot as plt
import numpy as np
from typing import Tuple
from heapq import heappush, heappop
import math

path = []  # global var


#dim=int(input("What is the dimension of the maze?\n"))

class Coord:
    def __init__(self, m: {np.ndarray, list}, x: int, y: int):
        self.x = x
        self.y = y
        self.m = m
        self.dim = len(m)
        self.prev = None  # for tracing the path in DFS/BFS
        self.steps = 0
        self.distance = 0  # for A* search algorithm

    def is_valid(self):
        """check whether the Coord is within the maze.
        :return: boolean
        """
        return (self.x >= 0) and (self.x < self.dim) and (self.y >= 0) and (self.y < self.dim)

    def is_available(self):
        """check whether the Coord is not occupied and also not restricted
        :return: boolean
        """
        return self.is_valid() and self.m[self.x][self.y] == 0

    def restrict(self):
        """mark Coord as restricted.
        :return: void
        """
        self.m[self.x][self.y] = -1

    def get_neighbors(self):
        """get the neighbor cells for the given cell
        :param self: location (an integer tuple) of the cell
        :return: a list of available neighbor cells
        """
        m = self.m
        up = Coord(m, self.x - 1, self.y)
        down = Coord(m, self.x + 1, self.y)
        left = Coord(m, self.x, self.y - 1)
        right = Coord(m, self.x, self.y + 1)
        elements = [up.is_available() and up, left.is_available() and left,
                    down.is_available() and down, right.is_available() and right]
        return [e for e in elements if e]  # get rid of 'False' elements

    def update_distance(self):
        #  steps already taken + Euclidean distance to the goal
        self.distance = self.steps + math.sqrt((self.x - self.dim - 1) ** 2
                                               + (self.y - self.dim - 1) ** 2)

    def get_path(self):
        cell = self
        path = []
        while cell:
            path.insert(0, (cell.x, cell.y))  # insert to the front of the list
            cell = cell.prev
        return path

    def __lt__(self, other: 'Coord'):  # for comparison in priority heap
        return self.distance < other.distance


def is_reachable(m: {np.ndarray, list}, start: Tuple[int, int], goal: Tuple[int, int], is_dfs: bool):
    """check whether goal(goal) is reachable from s(start) in the maze
    :param m: maze
    :param start: int tuple contains starting coordinates
    :param goal: int tuple contains goal coordinates
    :param is_dfs: True/False for dfs/bfs
    :return whether start and goal are reachable from each other
    """
    if m[start[0]][start[1]] == 1 or m[goal[0]][goal[1]] == 1:
        return False  # not reachable because the start/target c is occupied.
    fringe = [Coord(m, start[0], start[1])]  # stack/queue
    while len(fringe) != 0:
        cell = fringe.pop() if is_dfs else fringe.pop(0)  # dfs/bfs
        if (cell.x, cell.y) == goal:
            global path
            path = cell.get_path()
            # print(cell.get_path())
            return True
        else:  # add valid neighbor cells to fringe
            neighbors = cell.get_neighbors()
            for ne in neighbors:
                ne.prev = cell
            fringe.extend(neighbors)
            cell.restrict()  # add restriction
    return False


def dfs(m: {np.ndarray, list}, start: Tuple[int, int], goal: Tuple[int, int]):
    return is_reachable(m, start, goal, True)


def bfs(m: {np.ndarray, list}, start: Tuple[int, int], goal: Tuple[int, int]):
    return is_reachable(m, start, goal, False)


def distance_priority_search(m: {np.ndarray, list}, start: Tuple[int, int], goal: Tuple[int, int]):
    """
    A* search: find the shortest path from start to goal based on Euclidean distance to the goal
    :param m: maze
    :param start: int tuple contains starting coordinates
    :param goal: int tuple contains goal coordinates
    :return whether start and goal are reachable from each other
    """
    if m[start[0]][start[1]] == 1 or m[goal[0]][goal[1]] == 1:
        return False  # not reachable because the start/target cell is occupied.
    heap = []  # priority heap
    heappush(heap, Coord(m, start[0], start[1]))
    while len(heap) != 0:
        cell = heappop(heap)
        cell.steps += 1
        cell.update_distance()
        if (cell.x, cell.y) == goal:
            global path
            path = cell.get_path()
            # print(len(path), path, cell.steps, cell.distance)
            return True
        else:  # add valid neighbor cells to fringe
            neighbors = cell.get_neighbors()
            for ne in neighbors:
                ne.prev = cell
                ne.steps = cell.steps
                ne.update_distance()
                heappush(heap, ne)
            cell.restrict()  # add restriction
    return False

def generate_maze(dim, p):
    mat = [[1 if random.uniform(0,1) <= p else 0 for j in range(dim)] for i in range(dim)]
    mat[0][0] = 0
    mat[dim-1][dim-1] = 0
    plt.imshow(mat, cmap='Greys', interpolation="nearest")
    plt.show()
    return mat


# def DFS(maze, src=Coord(0,0), dest=Coord(dim, dim)):
#     print(maze[src.x])

# def problem_1(p, dim=dim):
#     print('Here is your maze:')
#     return generate_maze(dim, p)

# maze = problem_1(0.3)
# DFS(maze)


def test_dfs(dim: int, trials: int):
    results = []
    density = 0
    while density <= 0.6:
        trial = 0
        res = []
        while trial < trials:
            mat = generate_maze(dim, density)
            res.append(dfs(mat, (0, 0), (dim - 1, dim - 1)))
            trial += 1
        print(f'density={round(density, 3)}, res={res.count(True)}')
        results.append((round(density, 3), 0.01 * res.count(True)))
        density += 0.01
    print(results)


if __name__ == '__main__':
    maze = generate_maze(10, 0.2)
    print(distance_priority_search(maze, (0, 0), (9, 9)))
    print(len(path), path)
