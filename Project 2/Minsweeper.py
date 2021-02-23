import numpy as np
import random
from typing import Tuple

class MinesweeperGame:
    def __init__(self, d:int, n:int):
        self._dim = d
        self._n = n
        self._board = np.zeros([d, d], dtype=int)
    
    def _get_neighbors(self, cell: Tuple[int, int]):
        x, y = cell
        transform = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1), (x + 1 , y + 1), (x + 1, y - 1), (x - 1, y - 1), (x - 1, y + 1)]
        return [(a, b) for (a, b) in transform if 0 <= a < self._dim and 0 <= b < self._dim]
        
    def _add_mines(self):
        for num in range(self._n):
            x = random.randint(0, self._dim-1)
            y = random.randint(0, self._dim-1)
            self._board[x][y] = -1
            
            neighbors = self._get_neighbors((x,y))
            for neighbor in neighbors:
                if self._board[neighbor[0]][neighbor[1]] != -1:
                    self._board[neighbor[0]][neighbor[1]] += 1
                    
    def start_game(self):
        self._add_mines()
        print(self._board)
        
    def query(self, cell: Tuple[int, int]):
        return self._board[cell[0]][cell[1]]
        
if __name__ == '__main__':
    game = MinesweeperGame(10, 10)
    game.start_game()