import sys
import time

from Minesweeper import MinesweeperGame
import random
import numpy as np
from Agent import Agent as ImprovedAgent
from Environment import Environment


class Cell:
    def __init__(self, x: int, y: int, d: int):
        self.x = x
        self.y = y
        self.value = -2
        self.safe = 0
        self.mines = 0
        self.hidden = 0
        self._dim = d

    def get_neighbors(self):
        x, y = self.x, self.y
        transform = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1), (x + 1, y + 1), (x + 1, y - 1), (x - 1, y - 1),
                     (x - 1, y + 1)]
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
    def __init__(self, d: int, game: MinesweeperGame):
        self._game = game
        self._board = [[Cell(row, col, d) for col in range(d)] for row in range(d)]
        self._dim = d
        self._to_check = []
        self._identified = 0

    def play_game(self):
        while len(self.get_hidden()) > 0:
            rand_x, rand_y = self.pick_random()
            rand_cell = self._board[rand_x][rand_y]
            rand_cell.value = self._game.query((rand_x, rand_y))
            rand_cell.update_info(self._board)
            self._to_check.append(rand_cell)
            # print("Random", rand_cell.x, rand_cell.y)
            self.print_board()
            free = self.get_free()
            for cell in free:
                if self._board[cell[0]][cell[1]] not in self._to_check:
                    self._to_check.append(self._board[cell[0]][cell[1]])
                    self._board[cell[0]][cell[1]].update_info(self._board)
            i = 0
            while i < len(self._to_check):
                cell = self._to_check[i]
                if cell.value == -1:
                    self._to_check.remove(cell)
                    continue
                cell.update_info(self._board)
                if cell.hidden == 0:
                    self._to_check.remove(cell)
                    continue
                neighbors = cell.get_neighbors()
                if (cell.value - cell.mines) == cell.hidden:
                    for neighbor in neighbors:
                        x, y = neighbor[0], neighbor[1]
                        if self._board[x][y].value == -2:
                            self._board[x][y].value = -1
                            self._board[x][y].update_info(self._board)
                            self._identified += 1
                    for neighbor in neighbors:
                        x, y = neighbor[0], neighbor[1]
                        self._board[x][y].update_info(self._board)
                    cell.update_info(self._board)
                    free = self.get_free()
                    for cells in free:
                        if self._board[cells[0]][cells[1]] not in self._to_check:
                            self._to_check.append(self._board[cells[0]][cells[1]])
                            self._board[cells[0]][cells[1]].update_info(self._board)
                elif ((len(neighbors) - cell.value) - cell.safe) == cell.hidden:
                    for neighbor in neighbors:
                        x, y = neighbor[0], neighbor[1]
                        if self._board[x][y].value == -2:
                            self._board[x][y].value = self._game.query((x, y))
                            self._board[x][y].update_info(self._board)
                    for neighbor in neighbors:
                        x, y = neighbor[0], neighbor[1]
                        self._board[x][y].update_info(self._board)
                    cell.update_info(self._board)
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
                # print(cell.x, cell.y)
                self.print_board()
                if len(self.get_hidden()) == 0:
                    break
        return self._identified

    def print_board(self):
        board_vals = np.empty([self._dim, self._dim], dtype=int)
        for row in range(self._dim):
            for col in range(self._dim):
                board_vals[row][col] = self._board[row][col].value
        # print(board_vals)

    def pick_random(self):
        hidden = self.get_hidden()
        index = random.randint(0, len(hidden) - 1)
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
    print(f'System Recursion Limit: {sys.getrecursionlimit()}')
    limit = 100000000
    print(f'Change system recursion limit to {limit}')
    sys.setrecursionlimit(limit)

    dim = 10
    mines = 5
    trials = 30
    stats_basic = []
    times_1 = []
    stats_random = []
    times_2 = []
    stats_im_backtrack = []
    times_3 = []
    # stats_global_info = []
    while mines <= 90:
        rates = []
        time_basic = []
        rates_random = []
        time_random = []
        rates_im_backtrack = []
        time_backtrack = []
        # rates_info = []
        for i in range(trials):
            print(f'{mines} - ({i + 1}/{trials})')
            game = MinesweeperGame(dim, mines)
            game.start_game()  # This is how we will test both agents, so that both agents do operations on same game for consistency
            agent = Agent(dim, game)
            start_time = time.time()
            rate = agent.play_game() / mines
            end_time = time.time()
            time_basic.append(end_time - start_time)
            rates.append(rate)
            print(f'\t Basic    : {rate}')
            board = game.get_board()
            board[board != -1] = 0
            board[board == -1] = 1
            #
            env_1 = Environment(dim, mines, board)
            agent_random = ImprovedAgent(env_1, True)
            start_time = time.time()
            fails, coord = agent_random.begin()
            end_time = time.time()
            time_random.append(end_time - start_time)
            rate_random = (mines - fails) / mines
            rates_random.append(rate_random)
            print(f'\t Random    : {rate_random}')

            env_2 = Environment(dim, mines, board)
            im_agent_backtrack = ImprovedAgent(env_2, False)
            start_time = time.time()
            fails, _ = im_agent_backtrack.begin(coord)
            end_time = time.time()
            time_backtrack.append(end_time - start_time)
            rate_im_backtrack = (mines - fails) / mines
            rates_im_backtrack.append(rate_im_backtrack)
            print(f'\t Backtrack : {rate_im_backtrack}')
            #
            # env_3 = Environment(dim, mines, board)
            # agent_info = ImprovedAgent(env_3, False, mines)
            # successes, _ = agent_info.begin()
            # rate_info = successes / mines
            # rates_info.append(rate_info)
            # print(f'\t Info      : {rate_info}')
        mines += 5
        stats_basic.append(sum(rates) / len(rates))
        times_1.append(sum(time_basic) / len(time_basic))
        stats_random.append(sum(rates_random) / len(rates_random))
        times_2.append(sum(time_random) / len(time_random))
        stats_im_backtrack.append(sum(rates_im_backtrack) / len(rates_im_backtrack))
        times_3.append(sum(time_backtrack) / len(time_backtrack))
        # stats_global_info.append(sum(rates_info) / len(rates_info))
    print(stats_basic)
    print(stats_random)
    print(stats_im_backtrack)
    print(times_1)
    print(times_2)
    print(times_3)
    # print(stats_global_info)

# if __name__ == '__main__':
#     dim = 5
#     mines = 10
#     env_1 = Environment(dim, mines)
#     agent_random = ImprovedAgent(env_1, True)
#     fails, coord = agent_random.begin()
#     rate_random = (mines - fails) / mines
#     print(f'Success Rate: {rate_random}')
