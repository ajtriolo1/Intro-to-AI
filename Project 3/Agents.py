import random
from Environment import Environment as Env
import numpy as np


class Agent:
    def __init__(self, en: Env):
        self.en = en
        self.type_map = en.get_map()
        self.dim = len(self.type_map)
        self.belief = np.full((self.dim, self.dim), 1 / (self.dim ** 2))
        self.fnr = {1: 1 / 10, 2: 3 / 10, 3: 7 / 10, 4: 9 / 10}
        self.fnr_map = np.vectorize(self.fnr.__getitem__)(self.type_map)
        # self.improved_strategy_1 = {1: 1, 2: 2, 3: 3, 4: 4}
        # self.improved_strategy_2 = {1: 2, 2: 2, 3: 3, 4: 3}

    def run(self, basic_strategy: {1, 2}, is_improved: bool):
        dim, belief = self.dim, self.belief
        searches, distance = 0, 0
        indices = np.random.randint(0, high=dim, size=2)  # choose 2 indices randomly
        i, j = indices[0], indices[1]
        if self.en.is_target((i, j)):
            searches += 1
            return searches, distance
        else:
            num_search = 1
            while True:
                fn = self.fnr_map[i][j]  # get negative false rate for cell_ij
                denominator = belief[i][j] * (fn ** num_search) + 1 - belief[i][j]
                for x in range(dim):
                    for y in range(dim):
                        numerator = belief[x][y] * (fn ** num_search) if x == i and y == j else belief[x][y]
                        belief[x][y] = numerator / denominator

                cells = []  # filter out cells with the highest belief and lowest distance
                if basic_strategy == 1:
                    cells = np.asarray(np.where(belief == np.max(belief))).T
                else:
                    confidence = np.multiply(belief, 1 - self.fnr_map)
                    cells = np.asarray(np.where(confidence == np.max(confidence))).T

                min_dist = 2 * dim
                dist_list = []
                for x in range(len(cells)):
                    a, b = cells[x]
                    dist = abs(a - i) + abs(b - j)
                    min_dist = min(min_dist, dist)
                    dist_list.append(dist)
                filtered_cells = [cells[x] for x in range(len(cells)) if dist_list[x] == min_dist]

                i, j = random.choice(filtered_cells)  # update the index of the cell to be searched
                if is_improved:
                    num_search = self.type_map[i][j]
                searches += num_search
                distance += min_dist
                for _ in range(num_search):
                    if self.en.is_target((i, j)):
                        print(
                            f'Found: searches: {searches}, distance: {distance}')
                        return searches, distance
