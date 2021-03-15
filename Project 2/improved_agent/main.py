import sys
from Environment import Environment, rand_mine
from Agent import Agent


if __name__ == '__main__':
    print(f'System Recursion Limit: {sys.getrecursionlimit()}')
    limit = 100000000
    print(f'Change system recursion limit to {limit}')
    sys.setrecursionlimit(limit)

    stats = []
    dim = 10
    mines = 5
    trials = 100
    while mines <= 90:
        rates = []
        for i in range(trials):
            matrix = rand_mine(dim, mines)
            env = Environment(dim, mines, matrix)
            agent = Agent(env)
            rate = agent.begin() / mines
            print(f'{mines}({i + 1}/{trials}): {rate}')
            rates.append(rate)
        print(rates)
        print(sum(rates) / len(rates))
        stats.append(sum(rates) / len(rates))
        mines += 5
    print(stats)
    # for max-search-depth = 10, dimension = 10:
    # mines =         [5,    10,  15,  20,  25,  30,  35,  40,  45,  50,  55,  60,  65,  70,  75, 80,  85,  90]
    # success_rates = [.98, .98, .96, .94, .94, .92, .85, .84, .83, .79, .78, .75, .68, .61, .6, .58, .56, .53]
