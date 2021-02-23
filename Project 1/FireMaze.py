import time
from Maze import *

"""Module for simulating fire for SAS"""
class FireMaze(MazeGame):
    def __init__(self, dim: int, p: float, q: float, maze, fire_coord):
        super().__init__(dim, p, q)
        self._dim = dim
        self._q = q
        self._matrix = maze # Maze to do simulations on
        self._fire_matrix = np.zeros([dim, dim])
        self._first_fire_coord = fire_coord # Initial fire position

    def get_transformed_coords(self, coord: Tuple[int, int]):
        x, y = coord
        transformed = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [(a, b) for (a, b) in transformed if 0 <= a < self._dim and 0 <= b < self._dim and
                self._matrix[a][b] != 1]

    def simulate_fire(self, trials: int):
        self._fire_matrix[self._first_fire_coord[0]][self._first_fire_coord[1]] = 1
        time_start = time.time()
        simulated_matrices = []
        i = 0
        while i < trials:
            matrix = self._matrix.copy()
            fire_record_matrix = self._fire_matrix.copy()
            coords_on_fire = [self._first_fire_coord]
            n = 2
            while True:
                fire_coords_with_available_neighbors = []  # store firing cells for the next round
                # sum of available neighbor cells (of currently fires) that are not on fire
                sum_potential_firing_neighbors = 0
                for coord in coords_on_fire:
                    transformed_coords = self.get_transformed_coords(coord)  # get adjacent coords
                    available_coords = [(a, b) for (a, b) in transformed_coords if matrix[a][b] == 0]  # filter coords
                    # if no potential firing neighbor, this coord won't proceed to the next round
                    if len(available_coords) != 0:
                        fire_coords_with_available_neighbors.append(coord)
                    for curr_coord in available_coords:
                        coords_to_check = self.get_transformed_coords(curr_coord)
                        # count the numbers of firing neighbors
                        k = sum(1 for curr_coord in coords_to_check if
                                matrix[curr_coord[0]][curr_coord[1]] == 3)
                        sum_potential_firing_neighbors += k
                        curr_fire_prob = 1 - (1 - self._q) ** k  # probability of the current neighbor to be on fire.
                        if random.uniform(0, 1) <= curr_fire_prob:
                            x, y = curr_coord
                            # the initial firing cell is marked as 1, so the following fire cells are marked by n in
                            # the nth round.
                            matrix[x][y] = 3
                            fire_record_matrix[x][y] = n
                            fire_coords_with_available_neighbors.append(curr_coord)
                if sum_potential_firing_neighbors == 0:  # not more cells to be on fire, simulation complete
                    break
                coords_on_fire = fire_coords_with_available_neighbors  # replace the firing list for the next round
                n += 1
            i += 1
            # the ith simulation ends, add the fire record matrix to the list
            simulated_matrices.append(fire_record_matrix)
        self._fire_matrix = np.nanmean(simulated_matrices, axis=0)  # get the average of the fire matrices.
        time_end = time.time()
        return self._fire_matrix
