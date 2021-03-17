from typing import Optional, Tuple
import numpy as np


transitions = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]


def rand_mine(d: int, n: int) -> np.ndarray:
    """
    generate a d*d binary mine_matrix to represent the game
    :param d: side length of the grid
    :param n: number of mines
    :return: a binary mine_matrix, 0 indicates safe square and 1 means mine
    """
    arr = np.array([0] * (d ** 2 - n) + [1] * n)  # d^2 - n zeros and n ones.
    np.random.shuffle(arr)  # shuffle the 1d array to have zeros and ones distributed randomly.
    matrix = np.reshape(arr, (-1, d))  # convert the 1d array to a square _matrix
    print("Board:")
    print(matrix)
    return matrix


class Environment:
    def __init__(self, d: int, n: int, matrix: Optional[np.ndarray] = None):
        self.dim = d
        self._matrix = matrix if matrix is not None else rand_mine(d, n)
        self._total_mines = n
        self._mines_discovered = 0
        self._incorrect_marks = 0
        self._query_list = []

    def query(self, x: int, y: int, first_query: Optional[bool] = False) -> (bool, int):
        """
        In responding to a query, the environment should specify whether or not there was a mine there,
        and if not, how many surrounding cells have mines.
        :param first_query: the first queries does not count into statictics
        :param x: coordinate-x
        :param y: coordinate-y
        :return: (False, n) for safe cell, (True, 0) for mine
        """
        if (x, y) not in self._query_list:
            if not first_query:
                self._query_list.append((x, y))
        if self._matrix[x][y] == 0:
            # return the amount of mines around the loc
            n = sum(0 <= x + dx < self.dim and 0 <= y + dy < self.dim and self._matrix[x + dx][y + dy] == 1
                    for (dx, dy) in transitions)
            return False, n  # is a safe cell, and its number of neighbor mines
        else:
            return True, 0   # is a mine

    def mark(self, x: int, y: int, val: int) -> bool:
        if self._matrix[x][y] == val:
            if val == 1 and (x, y) not in self._query_list:
                self._mines_discovered += 1
            return True
        self._incorrect_marks += 1
        return False
