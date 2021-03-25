import operator
import numpy as np
import random
from typing import Tuple, Optional, List, Dict
from Improved_Environment import transitions, Environment
from sympy import Matrix


def get_matrix(dim: int):
    """Square mine_matrix filled with -1"""
    arr = np.array([-1] * (dim ** 2))
    return np.reshape(arr, (-1, dim))


class Agent:
    def __init__(self, env: Environment, is_random_guess: Optional[bool] = False, mines: Optional[int] = -1, ):
        self.env = env
        self.dim = env.dim
        self.kb = []  # knowledge base
        self.mine_matrix = get_matrix(env.dim)  # 1 -> mine; 0 -> covered; -1 -> unknown
        self.num_matrix = get_matrix(env.dim)  # 0 to 8 -> number of neighbor mines; -1 -> unknown
        self.fail = 0
        self.is_random_guess = is_random_guess
        self.mines = mines

    def get_neighbor_vars(self, x: int, y: int):
        var_list = [(x + dx, y + dy) for (dx, dy) in transitions
                    if 0 <= x + dx < self.dim and 0 <= y + dy < self.dim and self.mine_matrix[x + dx][y + dy] == -1]
        return var_list

    def get_neighbor_coords(self, x: int, y: int):
        return [(x + dx, y + dy) for (dx, dy) in transitions if 0 <= x + dx < self.dim and 0 <= y + dy < self.dim]

    def begin(self, coord: Optional[Tuple[int, int]] = None):
        """ begin the game by the given coordination or by picking a cell randomly.
        :param coord: coordination to begin the game
        :return: number of successful guess of mines
        """
        # generate random coord, or to find the cell which is least likely to be a mine, if initial coord is not given
        x = random.choice(range(self.dim)) if coord is None else coord[0]
        y = random.choice(range(self.dim)) if coord is None else coord[1]
        is_mine, num = self.env.query(x, y, True)
        # print(x, y, is_mine, num)
        if is_mine:  # regenerate random coord until a covered cell is found
            if coord is None:
                self.begin()
        else:
            if coord is None:
                coord = (x, y)
            self.mine_matrix[x][y] = 0
            self.num_matrix[x][y] = num  # number of mines around (x, y)
            self.kb.append((x, y, num))
            self.reasoning()

        return self.fail, coord

    def proceed(self):
        """:return: whether the game is finished"""
        return (self.mine_matrix == -1).sum() > 0

    def reasoning(self):
        """ generate an augmented matrix and its RREF from the knowledge base to find new knowledge.
        :return: None
        """
        if not self.proceed():
            return
        filtered_kb = []
        var_list = []
        for (x, y, v) in self.kb:
            neighbor_list = self.get_neighbor_vars(x, y)
            for (i, j) in neighbor_list:
                if self.mine_matrix[i][j] == -1:  # has unknown neighbor cells
                    if (x, y, v) not in filtered_kb:
                        filtered_kb.append((x, y, v))
                    if (i, j) not in var_list:
                        var_list.append((i, j))
        self.kb = filtered_kb
        if len(self.kb) == 0:
            self._random_guess()

        # the last column stores the value for each equation
        matrix = np.zeros((len(filtered_kb), len(var_list) + 1), dtype='i')  # integer data type

        row = 0
        for (x, y, v) in filtered_kb:
            num = v
            for (i, j) in self.get_neighbor_coords(x, y):
                # print(f'Checking: {(x, y, v)}: {(i, j)} = {self.mine_matrix[i][j]}')
                if self.mine_matrix[i][j] != -1:  # is not a var, decrease # of unknown neighbor cells
                    # print(f'{i},{j} is known!!!')
                    num -= self.mine_matrix[i][j]
                else:  # find col index of the current cell in var_list and update the matrix
                    col = var_list.index((i, j))
                    matrix[row][col] = 1
            matrix[row][-1] = num  # add remaining sum to the last column of the current row
            # print(f'v: {v}; num: {num}')
            row += 1

        #self.print_kb()
        #print(f'Variable list: {var_list}')
        #print(f'Augmented matrix (equations):')
        #print(matrix)
        has_new_info_1 = self.evaluate_equations(var_list, matrix)
        #print(f'RREF of matrix:')
        rref_matrix = Matrix(matrix).rref()[0].tolist()
        #print(rref_matrix)
        has_new_info_2 = self.evaluate_equations(var_list, rref_matrix)

        has_new_info = has_new_info_1 or has_new_info_2

        if self.proceed():
            if has_new_info:
                # if new knowledge was produced, mark game map, update kb and regenerate/redo matrix/RREF.
                if len(self.kb) != 0:
                    self.reasoning()
                else:
                    self._random_guess()
            else:
                if len(self.kb) == 0:
                    self._random_guess()
                else:
                    if self.is_random_guess:
                        self._random_guess()
                    else:
                        guess = Guess(var_list, matrix)
                        ((x, y), num), _ = guess.backtrack()
                        self.update((x, y), num)

                if len(self.kb) != 0:
                    self.reasoning()
                else:
                    if self.proceed():
                        self._random_guess()

    def _random_guess(self):
        """ if there is nothing in the knowledge base, this function is called.
        :return: None
        """
        x, y = np.where(self.mine_matrix == -1)
        i = np.random.randint(len(x))
        random_pos = (x[i], y[i])
        # print
        #print(f'Random Guess: {random_pos} is safe')
        self.update(random_pos, 0)
        if len(self.kb) != 0:
            self.reasoning()
        else:
            if self.proceed():
                self._random_guess()

    def print_kb(self):
        mines = self.mine_matrix.copy()
        print(str(mines).replace('-1', '  '))
        nums = self.num_matrix.copy()
        print(str(nums).replace('-1', '  '))
        print(f'Knowledge base: {self.kb}')

    def evaluate_equations(self, var_list: List[Tuple[int, int]], equations: {list, np.ndarray}) -> bool:
        """ looking for trivial solutions in equations.
        :param var_list: a list of variables corresponds to the coefficients of the equations(matrix).
        :param equations: an augmented matrix to represents the constraints.
        :return:
        """
        has_new_info = False
        for i in range(len(equations)):
            positive_var_indices = [k for k in range(len(equations[i]) - 1) if equations[i][k] == 1]
            negative_var_indices = [k for k in range(len(equations[i]) - 1) if equations[i][k] == -1]
            value = equations[i][-1]

            len_p = len(positive_var_indices)
            len_n = len(negative_var_indices)
            # case 1: all non-zero variables are with the same sign and right hand value is 0: all vars are 0.
            if ((len_p == 0 and len_n != 0) or (len_p != 0 and len_n == 0)) and value == 0:
                has_new_info = True
                for index in positive_var_indices + negative_var_indices:
                    # add to knowledge base; mark the cell; update num and mine matrices.
                    # print
                    #print(f"Inference: {var_list[index]} is safe")
                    self.update(var_list[index], 0)
                continue
            # case 2: all non-zero variables are with the same sign and sum(left) = value: all vars are 1.
            if ((len_p != 0 and len_n == 0) and value == len_p) or ((len_p == 0 and len_n != 0) and value == 0 - len_n):
                has_new_info = True
                for index in positive_var_indices + negative_var_indices:
                    # self.update(var_list[index], 1)
                    # print
                    #print(f"Inference: {var_list[index]} is a mine")
                    x, y = var_list[index]
                    self.mine_matrix[x][y] = 1
                continue
            # case 3: a + b - c = -1 : a = b = 0, c = 1
            if len_n == 1 and value == -1:
                has_new_info = True
                for index in positive_var_indices:
                    # print
                    #print(f"Inference: {var_list[index]} is safe")
                    self.update(var_list[index], 0)
                for index in negative_var_indices:
                    # self.update(var_list[index], 1)
                    # print
                    #print(f"Inference: {var_list[index]} is a mine")
                    x, y = var_list[index]
                    self.mine_matrix[x][y] = 1
                continue
            # case 4: a - b - c = 1: a = 1, b = c = 0
            if len_p == 1 and value == 1:
                has_new_info = True
                for index in positive_var_indices:
                    # self.update(var_list[index], 1)
                    # print
                    #print(f"Inference: {var_list[index]} is a mine")
                    x, y = var_list[index]
                    self.mine_matrix[x][y] = 1
                for index in negative_var_indices:
                    # print
                    #print(f"Inference: {var_list[index]} is safe")
                    self.update(var_list[index], 0)
                continue

        return has_new_info

    def update(self, coord: Tuple[int, int], val: {0}):
        """ mark cells based on inference or guess.
        :param coord: coordination of the cell to be marked
        :param val: 0 or 1
        :return: whether the mark is correct
        """
        x, y = coord
        if self.mine_matrix[x][y] == -1:
            if self.env.mark(x, y, val):
                #print(f'Mark ({x}, {y}) as safe')
                self.mine_matrix[x][y] = 0
                _, mines = self.env.query(x, y)
                self.kb.append((x, y, mines))
                self.num_matrix[x][y] = mines
            else:
                self.mine_matrix[x][y] = 1
                self.fail += 1
                #print(f'Error: Mark ({x}, {y}) as safe')


class Guess:
    def __init__(self, var_list: list, equations: {list, np.ndarray}):
        self.var_list = var_list
        self.equations = equations
        self.max_var_index = 10

    def get_priority_list(self):
        """ order the variables in a list based on their appearances in the knowledge base.
        :return: None
        """
        frequency = [(0, (x, y)) for (x, y) in self.var_list]
        for i in range(len(self.equations)):
            for j in range(len(self.equations[i]) - 1):  # the last column is the right hand side value.
                if self.equations[i][j] != 0:
                    k, (x, y) = frequency[j]  # k is the current frequency of var (x, y)
                    frequency[j] = (k + 1, (x, y))
        frequency.sort(reverse=True)  # descending order based on their appearances in the equations.
        sorted_vars = [(x, y) for _, (x, y) in frequency]
        return sorted_vars

    def backtrack(self) -> (((int, int), int), float):
        """ backtracking search to calculate the probability of unknown variables to be 1 or 0
        :return: the variable with its most possible value, along with the probability.
        """
        initial_var_index = 0
        record = (0, 0)  # store successes of the left and right subtree
        priorities = self.get_priority_list()
        prob_dict = {}
        for i in range(min(len(self.var_list), self.max_var_index + 1)):
            list_1 = priorities[: i]
            list_2 = priorities[i + 1:]
            priority_list = [priorities[i]] + list_1 + list_2
            values = [1] * len(self.var_list)  # initial values for the variables.
            result = self.traversal(priority_list, initial_var_index, values, record)
            x, y = result  # x and y are the nums of successes for the 1st var to be 1 and 0.
            # P(x=1 | equations) = x / (x + y)
            if x + y != 0:
                rate_is_safe = y / (x + y)
                # prob_dict[(priority_list[0], 1)] = rate_is_mine
                # prob_dict[(priority_list[0], 0)] = 1 - rate_is_mine  # P(x=0 | equations) = 1 - P(x=1 | equations)
                prob_dict[(priority_list[0], 0)] = rate_is_safe  # P(x=safe | equations) = 1 - P(x=mine | equations)
        candidate = max(prob_dict.items(), key=operator.itemgetter(1))[0]  # the most possible guess
        return candidate, prob_dict[candidate]

    def traversal(self, priorities: List[Tuple[int, int]],
                  var_index: int, values: list, record: Tuple[int, int]) -> Tuple[int, int]:
        """ traverse the search tree, if any node is unsatisfiable, stop searching its child nodes.
        :param priorities: list of variables ordered from most constrained to least constrained.
        :param var_index: the index of the last variable which is given a value.
        :param values: values corresponds to the variables in the priority list.
        :param record: (# of successes for parent node = 1, # of failures for parent node = 0)
        :return: record
        """
        var_val_dict = {}
        for i in range(var_index + 1):
            var_val_dict[priorities[i]] = values[i]
        is_leaf_node = var_index == len(self.var_list) - 1
        is_local_search = self.max_var_index < len(self.var_list) - 1

        satisfiable = self.is_satisfiable(var_val_dict, is_leaf_node, is_local_search)
        if satisfiable:
            if is_leaf_node or var_index == self.max_var_index:
                x, y = record
                if values[0] == 1:
                    x += 1
                else:
                    y += 1
                record = (x, y)
                # go back to parent node with value of 1
                j = var_index
                while j >= 0:
                    if values[j] != 0:
                        break
                    else:
                        j -= 1
                if j == -1:
                    return record  # no more node to explore, recursion finished.
                else:
                    var_index = j
                    values[var_index] = 0

            else:  # go to the child node
                var_index += 1
                values[var_index] = 1

        else:  # not satisfiable, go to its parent node with value of 1
            j = var_index
            while j >= 0:
                if values[j] != 0:
                    break
                else:
                    j -= 1
            if j == -1:
                return record  # no more node to explore, recursion finished.
            else:
                var_index = j
                values[var_index] = 0

        return self.traversal(priorities, var_index, values, record)

    def is_satisfiable(self, var_val_dict: Dict[Tuple[int, int], int],
                       is_leaf_node: bool, is_local_search: bool) -> bool:
        """ check whether the given values for variables are satisfiable
        :param var_val_dict: (key = variable, value = given value)
        :param is_leaf_node: whether the dictionary contains the variable which is the lead node of the backtrack tree
        :param is_local_search: local search or global search
        :return: whether variable-value paris are satisfiable in the knowledge base
        """
        equations = self.equations
        for i in range(len(equations)):
            sum_known_vars = 0
            for j in range(len(equations[i]) - 1):  # the last column is for right hand side values
                if equations[i][j] == 1 and self.var_list[j] in var_val_dict:
                    sum_known_vars += var_val_dict.get(self.var_list[j])
            if is_local_search:
                if sum_known_vars > equations[i][-1]:
                    return False
            else:
                if is_leaf_node:  # left hand side should be = the right hand side
                    if sum_known_vars != equations[i][-1]:
                        return False
                else:  # left hand side should be <= the right hand side
                    if sum_known_vars > equations[i][-1]:
                        return False
        return True
