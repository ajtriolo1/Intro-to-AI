import numpy as np
import matplotlib.pyplot as plt
from Environment import Environment as Env
from Improved_Agent import Agent


def test_agents():
    result = np.zeros([5, 5])
    # i = 4
    maps, trials_per_map = 10, 10
    # while i > 0:
    ave_cost_1 = []
    ave_cost_2 = []
    ave_cost_3 = []
    # ave_cost_4 = []
    for j in range(maps):
        en = Env(50)
        cost_1 = []
        cost_2 = []
        cost_3 = []
        # cost_4 = []
        for k in range(trials_per_map):
            print(f'map: {j + 1}/{maps}, play:  {k + 1}/{trials_per_map}')
            # en.set_target_on_type(i)
            en.set_target()
            en.print_target()

            # agent_1 = Agent(en)
            # searches_1, distance_1 = agent_1.run_improved()
            # sum_1 = searches_1 + distance_1
            # cost_1.append(sum_1)

            agent_1 = Agent(en)
            searches_1, distance_1 = agent_1.run(1, False)
            sum_1 = searches_1 + distance_1
            cost_1.append(sum_1)

            agent_2 = Agent(en)
            searches_2, distance_2 = agent_2.run(2, False)
            sum_2 = searches_2 + distance_2
            cost_2.append(sum_2)

            agent_3 = Agent(en)
            searches_3, distance_3 = agent_3.run_improved(10000)
            sum_3 = searches_3 + distance_3
            cost_3.append(sum_3)

            # agent_4 = Agent(en)
            # searches_4, distance_4 = agent_4.run(2, True)
            # sum_4 = searches_4 + distance_4
            # cost_4.append(sum_4)

        ave_cost_1.append(sum(cost_1) / len(cost_1))
        ave_cost_2.append(sum(cost_2) / len(cost_2))
        ave_cost_3.append(sum(cost_3) / len(cost_3))
        # ave_cost_4.append(sum(cost_4) / len(cost_4))
    result[0][1] = sum(ave_cost_1) / len(ave_cost_1)
    result[0][2] = sum(ave_cost_2) / len(ave_cost_2)
    result[0][3] = sum(ave_cost_3) / len(ave_cost_3)
    # result[i][4] = sum(ave_cost_4) / len(ave_cost_4)
    print(result)
    # i -= 1


def plot():
    # basic agents
    labels = ['flat', 'hill', 'forest', 'cave', 'average']
    agent_1 = [9647.74, 16453.87, 22478.89, 28071.17]
    agent_1.append(sum(agent_1) / len(agent_1))
    agent_2 = [3720.02, 7638.68, 15602.78, 40908.22]
    agent_2.append(sum(agent_2) / len(agent_2))
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, agent_1, width, label='Basic agent 1')
    rects2 = ax.bar(x + width / 2, agent_2, width, label='Basic agent 2')
    ax.set_ylabel('Scores')
    ax.set_title('Scores by target type and agent')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    fig.tight_layout()
    plt.show()

    # improved agent
    agents = ['Basic agent 1', 'Basic agent 2', 'Improved agent']
    y_pos = np.arange(len(agents))
    scores = [19081.42, 15120.45, 9140.33]
    plt.barh(y_pos, scores, align='center', alpha=0.7)
    plt.yticks(y_pos, agents)
    plt.xlabel('average score')
    plt.title('Average Score by Agent')
    plt.show()

# average scores of basic agent 1, basic agent 2, improved agent.
# 19081.42 15120.45  9140.33

# basic agent 1 and 2 on type 1, 2, 3, 4 targets
# 9647.74  3720.02
# 16453.87  7638.68
# 22478.89 15602.78
# 28071.17 40908.22
