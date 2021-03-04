import numpy as np
from typing import Tuple
from Minesweeper import *
import random

class Cell:
    def __init__(self, x:int, y:int, d:int):
        self.x = x
        self.y = y
        self.value = -2
        self.safe = 0
        self.mines = 0
        self.hidden = 0
        self._dim = d
        
    def get_neighbors(self):
        x, y = self.x, self.y
        transform = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1), (x + 1 , y + 1), (x + 1, y - 1), (x - 1, y - 1), (x - 1, y + 1)]
        return [(a, b) for (a, b) in transform if 0 <= a < self._dim and 0 <= b < self._dim]
    
    def update_info(self, game):
        neighbors = self.get_neighbors()
        hidden = 0
        mines = 0
        safe = 0
        for neighbor in neighbors:
            x, y = neighbor[0], neighbor[1]
            if game[x][y].value == -2:
                hidden += 1
            elif game[x][y].value == -1:
                mines += 1
            else:
                safe += 1
        self.hidden = hidden
        self.mines = mines
        self.safe = safe
                
class Agent:
    def __init__(self, d:int, n:int):
        self._game = MinesweeperGame(d, n)
        self._board  = [[Cell(row, col, d) for col in range(d)] for row in range(d)]
        self._dim = d
        self._to_check = []
        self._identified = 0
        self._mines = n
        
    def play_game(self):
        self._game.start_game()
        while len(self.get_hidden()) > 0:
            rand_x, rand_y = self.pick_random()
            rand_cell = self._board[rand_x][rand_y]
            rand_cell.value = self._game.query((rand_x, rand_y))
            rand_cell.update_info(self._board)
            self._to_check.append(rand_cell)
            print("Random", rand_cell.x, rand_cell.y)
            self.print_board()
            free = self.get_free()
            for cell in free:
                if self._board[cell[0]][cell[1]] not in self._to_check:
                    self._to_check.append(self._board[cell[0]][cell[1]])
                    self._board[cell[0]][cell[1]].update_info(self._board)
            i=0
            while i < len(self._to_check):
                cell = self._to_check[i]
                if cell.value == -1:
                    self._to_check.remove(cell)
                    continue
                cell.update_info(self._board)
                neighbors = cell.get_neighbors()
                if (cell.value - cell.mines) == cell.hidden:
                    for neighbor in neighbors:
                        x, y = neighbor[0], neighbor[1]
                        if self._board[x][y].value == -2:
                            self._board[x][y].value = self._game.query((x,y))
                            self._board[x][y].update_info(self._board)
                            self._identified += 1
                    for neighbor in neighbors:
                         x, y = neighbor[0], neighbor[1]
                         self._board[x][y].update_info(self._board)
                    free = self.get_free()
                    for cells in free:
                        if self._board[cells[0]][cells[1]] not in self._to_check:
                            self._to_check.append(self._board[cells[0]][cells[1]])
                            self._board[cells[0]][cells[1]].update_info(self._board)
                elif ((len(neighbors) - cell.value) - cell.safe) == cell.hidden:
                    for neighbor in neighbors:
                        x, y = neighbor[0], neighbor[1]
                        if self._board[x][y].value == -2:
                            self._board[x][y].value = self._game.query((x,y))
                            self._board[x][y].update_info(self._board)
                    for neighbor in neighbors:
                          x, y = neighbor[0], neighbor[1]
                          self._board[x][y].update_info(self._board)
                    free = self.get_free()
                    for cells in free:
                        if self._board[cells[0]][cells[1]] not in self._to_check:
                            self._to_check.append(self._board[cells[0]][cells[1]])
                            self._board[cells[0]][cells[1]].update_info(self._board)
                else:
                    cell.update_info(self._board)
                    self._to_check.remove(cell)
                    continue
                cell.update_info(self._board)
                self._to_check.remove(cell)
                print(cell.x, cell.y)
                self.print_board()
                if len(self.get_hidden()) == 0:
                    break
        print(self._identified/self._mines)

    
    def print_board(self):
        board_vals = np.empty([self._dim, self._dim], dtype=int)
        for row in range(self._dim):
            for col in range(self._dim):
                board_vals[row][col] = self._board[row][col].value
        print(board_vals)
    
    def pick_random(self):
        hidden = self.get_hidden()
        index  = random.randint(0, len(hidden)-1)
        return hidden[index]
    
    def get_hidden(self):
        board_vals = np.empty([self._dim, self._dim], dtype=int)
        for row in range(self._dim):
            for col in range(self._dim):
                board_vals[row][col] = self._board[row][col].value
        hidden = np.argwhere(board_vals == -2)
        return hidden
    
    def get_free(self):
        board_vals = np.empty([self._dim, self._dim], dtype=int)
        for row in range(self._dim):
            for col in range(self._dim):
                if self._board[row][col].hidden > 0:
                    board_vals[row][col] = self._board[row][col].value
                else:
                    board_vals[row][col] = -2
        free = np.argwhere(board_vals >= 0)
        return free.tolist()
            
if __name__ == '__main__':
    
    agent = Agent(10, 50)
    agent.play_game()