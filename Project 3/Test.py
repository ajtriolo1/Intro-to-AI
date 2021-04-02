import numpy as np

from Environment import Environment as Env
from Agents import Agent


def test_basic_agents():
    result = np.zeros([5, 5])
    i = 1
    maps, trials_per_map = 5, 5
    while i < 5:
        ave_cost_1 = []
        ave_cost_2 = []
        ave_cost_3 = []
        ave_cost_4 = []
        for j in range(maps):
            en = Env(50)
            cost_1 = []
            cost_2 = []
            cost_3 = []
            cost_4 = []
            for k in range(trials_per_map):
                print(f'Target type: {i}/4, map: {j + 1}/{maps}, play:  {k + 1}/{trials_per_map}')
                en.set_target_on_type(i)
                # en.set_target()
                en.print_target()

                agent_1 = Agent(en)
                searches_1, distance_1 = agent_1.run(1, False)
                sum_1 = searches_1 + distance_1
                cost_1.append(sum_1)

                agent_2 = Agent(en)
                searches_2, distance_2 = agent_2.run(2, False)
                sum_2 = searches_2 + distance_2
                cost_2.append(sum_2)

                agent_3 = Agent(en)
                searches_3, distance_3 = agent_3.run(1, True)
                sum_3 = searches_3 + distance_3
                cost_3.append(sum_3)

                agent_4 = Agent(en)
                searches_4, distance_4 = agent_4.run(2, True)
                sum_4 = searches_4 + distance_4
                cost_4.append(sum_4)

            ave_cost_1.append(sum(cost_1) / len(cost_1))
            ave_cost_2.append(sum(cost_2) / len(cost_2))
            ave_cost_3.append(sum(cost_3) / len(cost_3))
            ave_cost_4.append(sum(cost_4) / len(cost_4))
        result[i][1] = sum(ave_cost_1) / len(ave_cost_1)
        result[i][2] = sum(ave_cost_2) / len(ave_cost_2)
        result[i][3] = sum(ave_cost_3) / len(ave_cost_3)
        result[i][4] = sum(ave_cost_4) / len(ave_cost_4)
        print(result)
        i += 1
