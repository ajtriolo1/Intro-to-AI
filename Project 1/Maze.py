import random
from matplotlib import pyplot as plt

#dim=int(input("What is the dimension of the maze?\n"))

class Coord:
    def __init__(self, x:int, y:int):
        self.x=x
        self.y=y

def generate_maze(dim, p):
    mat = [[1 if random.uniform(0,1) <= p else 0 for j in range(dim)] for i in range(dim)]
    mat[0][0] = 0
    mat[dim-1][dim-1] = 0
    plt.imshow(mat, cmap='Greys', interpolation="nearest")
    plt.show()
    return mat

def isValid(row: int, col: int, dim: int):
    return (row >= 0) and (row < dim) and (col >= 0) and (col < dim)

def DFS(maze, src=Coord(0,0), dest=Coord(dim, dim)):
    print(maze[src.x])

def problem_1(p, dim=dim):
    print('Here is your maze:')
    return generate_maze(dim, p)

maze = problem_1(0.3)
DFS(maze)

