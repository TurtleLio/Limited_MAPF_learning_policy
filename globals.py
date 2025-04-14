import re
import os
import math
import json
import time
import heapq
import random
import pstats
import cProfile
import itertools
from itertools import combinations, permutations, tee, pairwise
from datetime import datetime
from typing import *
from collections import deque, defaultdict

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# GLOBAL OBJECTS
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

color_names = [
    # 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w',  # Single-letter abbreviations
    'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black',  # Full names
    'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'black',
    'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate',
    'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod',
    'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid',
    'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise',
    'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite',
    'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow',
    'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen',
    'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen',
    'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey',
    'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine',
    'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen',
    'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite',
    'navy', 'oldlace', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
    'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'red', 'rosybrown',
    'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue',
    'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet',
    'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen'
]

markers = [
    ".",    # point marker
    ",",    # pixel marker
    "o",    # circle marker
    "v",    # triangle_down marker
    "^",    # triangle_up marker
    "<",    # triangle_left marker
    ">",    # triangle_right marker
    "1",    # tri_down marker
    "2",    # tri_up marker
    "3",    # tri_left marker
    "4",    # tri_right marker
    "s",    # square marker
    "p",    # pentagon marker
    "*",    # star marker
    "h",    # hexagon1 marker
    "H",    # hexagon2 marker
    "+",    # plus marker
    "x",    # x marker
    "D",    # diamond marker
    "d",    # thin_diamond marker
    "P",    # plus (filled) marker
    "X",    # x (filled) marker
]
lines = [
    "-",  # solid line
    "--", # dashed line
    "-.", # dash-dot line
    ":",  # dotted line
]

markers_iter = iter(markers)
# markers_lines_dict = defaultdict(lambda: random.choice(markers))
markers_lines_dict = {}
colors_dict: DefaultDict[str, str | None] = defaultdict(lambda: None)


# markers_lines_dict['LNS2'] = '-p'
# colors_dict['LNS2'] = 'blue'
#
# markers_lines_dict['PF-LNS2'] = '-*'
# colors_dict['PF-LNS2'] = 'red'
#
# markers_lines_dict['PrP'] = '-v'
# colors_dict['PrP'] = 'green'
#
# markers_lines_dict['PF-PrP'] = '-^'
# colors_dict['PF-PrP'] = 'orange'

# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# GLOBAL CLASSES
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
class Node:
    def __init__(self, x: int, y: int, neighbours: List[str] | None = None):
        self.x: int = x
        self.y: int = y
        self.neighbours: List[str] = [] if neighbours is None else neighbours
        self.neighbours_nodes: List[Node] = []
        self.xy_name: str = f'{self.x}_{self.y}'

    @property
    def xy(self):
        return self.x, self.y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        return self.xy_name < other.xy_name

    def __gt__(self, other):
        return self.xy_name > other.xy_name

    def __hash__(self):
        return hash(self.xy_name)

    def __str__(self):
        return self.xy_name

    def __repr__(self):
        return self.xy_name


class AgentAlg:
    def __init__(self, num: int, start_node: Node, goal_node: Node):
        self.num = num
        self.name = f'agent_{num}'
        self.start_node: Node = start_node
        self.start_node_name: str = self.start_node.xy_name
        self.curr_node: Node = start_node
        # self.curr_node_name: str = self.curr_node.xy_name
        self.goal_node: Node | None = goal_node
        self.goal_node_name: str = self.goal_node.xy_name
        self.alt_goal_node: Node | None = None
        self.message: str = ''
        self.path: List[Node] | None = [self.start_node]
        self.k_path: List[Node] | None = [self.start_node]
        self.init_priority: float = random.random()
        self.priority: float = self.init_priority
        self.best_path_without_constraints : List[Node] | None = [self.start_node]

    @property
    def path_names(self):
        return [n.xy_name for n in self.path]

    def update_curr_node(self, i_time):
        if i_time >= len(self.path):
            self.curr_node = self.path[-1]
            return
        self.curr_node = self.path[i_time]

    def get_goal_node(self) -> Node:
        if self.alt_goal_node is not None:
            return self.alt_goal_node
        if self.goal_node is not None:
            return self.goal_node
        return self.curr_node

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __lt__(self, other: Self):
        return self.priority < other.priority

    def __hash__(self):
        return hash(self.num)

    def __eq__(self, other):
        return self.num == other.num

# class resource_distribution:
#     def __init__(self, max_nodes_for_agents: np.ndarray):
#         self.max_nodes = max_nodes_for_agents
# class resource_distribution:
#     def __init__(self, resources_per_agent: np.ndarray, num_agents: int, share_fun_name):
#         distrubated_resources = self.share_fun_name()
#         self.max_nodes = distrubated_resources
#         self.string_share_fun_ = str(share_fun_name)
#
#     def fixed_distribution(self):
#         # distributed_array = np.zeros(num_agents)
#         # total_amount = resources_per_agent*num_agents
#         # for i in range(num_agents):
#         #     distributed_array[i] = total_amount/num_agents
#         # return np.array(distributed_array)
#         total_amount = np.array([self.resources_per_agent] * self.num_agents)
#         return total_amount
#
#     def shared_distribution(self):
#         total_amount = np.array([self.resources_per_agent * self.num_agents] * self.num_agents)
#         return total_amount
#
#     def PIB_distribution(self):
#         precentage_of_shared_pool = 50
#         total_amount = self.resources_per_agent * self.num_agents
#         shared_pool = total_amount * precentage_of_shared_pool / 100
#         total_amount = np.array([self.resources_per_agent - int(shared_pool / self.num_agents)] * self.num_agents)
#         total_amount = np.append(total_amount, shared_pool)
#         return total_amount
def create_resource_class(base_distribution_class, neib_resources, neib_agents_distribution, class_name=None):
    if class_name is None:
        class_name = f"{base_distribution_class.distribution_name}-{neib_resources.neib_budget_name}-{neib_agents_distribution.neib_agents_name}"
    # Create a new class dynamically
    return type(class_name, (base_distribution_class,) + (neib_resources,) + (neib_agents_distribution,), {'class_name': class_name})

class resource_distribution_class:
    distribution_name: str
    def __init__(self, resources_per_agent=0, num_agents=0):
        self.max_nodes = 0
        self.resources_per_agent = resources_per_agent
        self.num_agents = num_agents
        self.distribution_name = None
        self.proportions = np.array([])
        self.agent_subset = []
        self.neib_budget = 0
    def input_parameters(self,resources_per_agent: int, num_agents: int):
        self.resources_per_agent = resources_per_agent
        self.num_agents = num_agents
    def resource_distribution(self):
        pass
    def resource_subtraction(self,index):
        pass
    def resource_collection(self):
        self.proportions = np.array([])
        resource_sum = 0
        for i, nodes_left in enumerate(self.max_nodes):
            # self.proportions.append(nodes_left)
            self.proportions= np.append(self.proportions, nodes_left)
            resource_sum += nodes_left
            self.max_nodes[i] = 0
        self.max_nodes = np.append(self.max_nodes,resource_sum)
        return self.proportions
    def resource_forward(self, agent : AgentAlg, mixed_agents_lis: List[AgentAlg]):
        return None

class fixed_distribution(resource_distribution_class):
    distribution_name = 'fixed'
    def __init__(self, resources_per_agent=0, num_agents=0):
        super().__init__(resources_per_agent, num_agents)
    def resource_distribuition(self):
        total_amount = np.array([self.resources_per_agent] * self.num_agents)
        self.max_nodes = total_amount
        return total_amount
    def resource_subtraction(self, index):
        if self.max_nodes[index] <= 0:
            return False
        self.max_nodes[index] -= 1
        return True
    def resource_forward(self, agent, mixed_agents_list):
        index = mixed_agents_list.index(agent)
        if index < (len(mixed_agents_list) - 1):
            next_agent = mixed_agents_list[index+1]
            # number_part_next_agent = next_agent.split("_")[1]
            # real_index_next_agent = int(number_part_next_agent)
            self.max_nodes[next_agent.num] += self.max_nodes[agent.num]
            self.max_nodes[agent.num] = 0
        #print(f"max_nodes = {self.max_nodes}")
        return None
    # def resource_forward(self, agent, mixed_agents_list):
    #     return None

class shared_distribution(resource_distribution_class):
    distribution_name = 'shared'
    def __init__(self, resources_per_agent=0, num_agents=0):
        super().__init__(resources_per_agent, num_agents)
    def resource_distribuition(self):
        total_amount = np.array([0] * self.num_agents)
        total_amount = np.append(total_amount,self.resources_per_agent* self.num_agents)
        self.max_nodes = total_amount
        return total_amount
    def resource_subtraction(self, index):
        if self.max_nodes[-1] <= 0:
            # print(f'not enough resources for agent {index}')
            return False
        self.max_nodes[-1] -= 1
        #print(f"resource left: {self.max_nodes[-1]}")
        return True
    def resource_forward(self, agent, mixed_agents_list):
        return None


class PIB_distribution(resource_distribution_class):
    distribution_name = 'PIB'
    def __init__(self, resources_per_agent=0, num_agents=0):
        super().__init__(resources_per_agent, num_agents)
    def resource_distribuition(self):
        precentage_of_shared_pool = 50
        total_amount = self.resources_per_agent * self.num_agents
        shared_pool = total_amount * precentage_of_shared_pool / 100
        total_amount = np.array([self.resources_per_agent - int(shared_pool / self.num_agents)] * self.num_agents)
        total_amount = np.append(total_amount, shared_pool)
        self.max_nodes = total_amount
        return total_amount
    def resource_subtraction(self, index):
        if self.max_nodes[-1] <= 0:
            #print(f'not enough resources for agent {index}')
            return False
        if self.max_nodes[index] != 0:
            self.max_nodes[index] -= 1
        else:
            self.max_nodes[-1] -= 1
            #print(f"agent index {index} took from shared pool | {self.max_nodes[-1]} left")
        return True


class neib_resources:
    neib_budget_name: str
    def neib_pool(self,neib_agents:[AgentAlg], fixed_neib_nodes: int=50, cp_graph: dict=[str, List[AgentAlg]], prefix: int = 5):
        pass

class neib_sum(neib_resources):
    # neib_budget_name = "neighbourhood-sum"
    neib_budget_name = "Sum"
    def neib_pool(self, neib_agents:[AgentAlg], fixed_neib_nodes: int=50, cp_graph: dict=[str, List[AgentAlg]], prefix: int = 5):
        self.agent_subset = neib_agents
        total_amount = 0
        for agent in neib_agents:
            total_amount += self.proportions[agent.num]
        if total_amount > self.max_nodes[-1]:
            total_amount = self.max_nodes[-1]
        self.neib_budget = total_amount
        self.max_nodes[-1] -= self.neib_budget
        #self.proportions = tuple(self.proportions)
        # print(f"this batch proportions: {np.array([self.proportions[agent.num] for agent in self.agent_subset])}")
        # print(f"total neighborhood budget {self.neib_budget} | left in bank: {self.max_nodes[-1]}")
        return self.neib_budget

class neib_proportions(neib_resources):
    neib_budget_name = "neighbourhood-proportions"
    def neib_pool(self, neib_agents:[AgentAlg], fixed_neib_nodes: int=50, cp_graph: dict=[str, List[AgentAlg]], prefix: int = 5):
        self.agent_subset = neib_agents
        #self.proportions = np.array(self.proportions)
        # if sum(self.proportions) == sum(self.max_nodes):
        #     self.proportions = [0]*self.num_agents
        #     sum_proportions = 0
        #     for agent, connections in cp_graph.items():
        #         agent_id = int(agent.split('_')[1])  # Extract the agent number (e.g., 55 from 'agent_55')
        #         #if len(connections) > 0:
        #         if len(connections) > 0 and len(connections) is not None:
        #             self.proportions[agent_id] = len(connections)
        #             sum_proportions += len(connections)
        #         else:
        #             #sum_proportions = 1
        #             #self.proportions[agent_id] = sum_proportions
        #             self.proportions[agent_id] = 0.01
        #     self.proportions = tuple(self.proportions)
        self.proportions = [0]*self.num_agents
        sum_proportions = 0
        for agent, connections in cp_graph.items():
            agent_id = int(agent.split('_')[1])  # Extract the agent number (e.g., 55 from 'agent_55')
            #if len(connections) > 0:
            if len(connections) > 0 and len(connections) is not None:
                self.proportions[agent_id] = len(connections)
                sum_proportions += len(connections)
            else:
                #sum_proportions = 1
                #self.proportions[agent_id] = sum_proportions
                self.proportions[agent_id] = 0.01
        self.proportions = tuple(self.proportions)
        # agents_index_arr = []
        # for agent in self.agent_subset:
        #     agents_index_arr.append(agent.num)
        for agent in neib_agents:
            #print(f"self.max_nodes[-1]:{self.max_nodes[-1]} | self.proportions[agent_num:{self.proportions[agent.num]} | sum_proportions:{sum(self.proportions)}")
            self.neib_budget += int(self.max_nodes[-1]*self.proportions[agent.num]/sum(self.proportions))
            #print(f"after math: {int(self.max_nodes[-1]*self.proportions[agent.num]/sum(self.proportions))}")
        if self.neib_budget <= len(self.agent_subset):
            self.neib_budget = len(self.agent_subset)
        if self.neib_budget > self.max_nodes[-1]:
            self.neib_budget = self.max_nodes[-1]
        self.max_nodes[-1] -= self.neib_budget
        #print(f"total amount for this batch:{self.neib_budget}")
        #print(f"this batch proportions: {np.array([self.proportions[agent.num] for agent in self.agent_subset])}")
        #print(f"resources in subset:{np.array([self.max_nodes[agent.num] for agent in neib_agents])} for agents:{np.array([agent.num for agent in neib_agents])}")
        return self.neib_budget

class neib_proportions_with_prefix(neib_resources):
    neib_budget_name = "Conflict-Proportional-Budget"
    def neib_pool(self, neib_agents:[AgentAlg], fixed_neib_nodes: int=50, cp_graph: dict=[str, List[AgentAlg]], prefix: int = 5):
        self.agent_subset = neib_agents
        #self.proportions = np.array(self.proportions)
        # if sum(self.proportions) == sum(self.max_nodes):
        #     self.proportions = [0]*self.num_agents
        #     sum_proportions = 0
        #     for agent, connections in cp_graph.items():
        #         agent_id = int(agent.split('_')[1])  # Extract the agent number (e.g., 55 from 'agent_55')
        #         #if len(connections) > 0:
        #         if len(connections) > 0 and len(connections) is not None:
        #             self.proportions[agent_id] = len(connections)
        #             sum_proportions += len(connections)
        #         else:
        #             #sum_proportions = 1
        #             #self.proportions[agent_id] = sum_proportions
        #             self.proportions[agent_id] = 0.01
        #     self.proportions = tuple(self.proportions)
        prefix_sub_limit = prefix*16
        self.proportions = [0]*self.num_agents
        sum_proportions = 0
        for agent, connections in cp_graph.items():
            agent_id = int(agent.split('_')[1])  # Extract the agent number (e.g., 55 from 'agent_55')
            #if len(connections) > 0:
            if len(connections) > 0 and len(connections) is not None:
                self.proportions[agent_id] = len(connections)
                sum_proportions += len(connections)
            else:
                #sum_proportions = 1
                #self.proportions[agent_id] = sum_proportions
                self.proportions[agent_id] = 0.01
        self.proportions = tuple(self.proportions)
        # agents_index_arr = []
        # for agent in self.agent_subset:
        #     agents_index_arr.append(agent.num)
        for agent in neib_agents:
            #print(f"self.max_nodes[-1]:{self.max_nodes[-1]} | self.proportions[agent_num:{self.proportions[agent.num]} | sum_proportions:{sum(self.proportions)}")
            self.neib_budget += int(self.max_nodes[-1]*self.proportions[agent.num]/sum(self.proportions))
            #print(f"after math: {int(self.max_nodes[-1]*self.proportions[agent.num]/sum(self.proportions))}")
        if self.neib_budget <= len(self.agent_subset):
            self.neib_budget = len(self.agent_subset)
        if self.neib_budget < prefix_sub_limit:
            self.neib_budget = prefix_sub_limit
        if self.neib_budget > self.max_nodes[-1]:
            self.neib_budget = self.max_nodes[-1]
        self.max_nodes[-1] -= self.neib_budget
        #print(f"total amount for this batch:{self.neib_budget}")
        #print(f"this batch proportions: {np.array([self.proportions[agent.num] for agent in self.agent_subset])}")
        #print(f"resources in subset:{np.array([self.max_nodes[agent.num] for agent in neib_agents])} for agents:{np.array([agent.num for agent in neib_agents])}")
        return self.neib_budget

class neib_fixed_50(neib_resources):
    # neib_budget_name = "neighbourhood-fixed-50"
    neib_budget_name = "Fixed-50"
    def neib_pool(self, neib_agents:[AgentAlg], fixed_neib_nodes: int=50, cp_graph: dict=[str, List[AgentAlg]], prefix: int = 5):
        self.agent_subset = neib_agents
        if self.max_nodes[-1] < fixed_neib_nodes:
            fixed_neib_nodes = self.max_nodes[-1]
        self.neib_budget = fixed_neib_nodes
        self.max_nodes[-1] -= self.neib_budget
        # if self.neib_budget < len(self.agent_subset):
        #     self.neib_budget = 0
        #print(f"total amount for this batch:{total_amount}")
        #print(f"resources in subset:{np.array([self.max_nodes[agent.num] for agent in neib_agents])} for agents:{np.array([agent.num for agent in neib_agents])}")
        return self.neib_budget

class neib_fixed_100(neib_resources):
    # neib_budget_name = "neighbourhood-fixed-100"
    neib_budget_name = "Fixed-100"
    def neib_pool(self, neib_agents: [AgentAlg], fixed_neib_nodes: int = 100,
                  cp_graph: dict = [str, List[AgentAlg]], prefix: int = 5):
        self.agent_subset = neib_agents
        if self.max_nodes[-1] < fixed_neib_nodes:
            fixed_neib_nodes = self.max_nodes[-1]
        self.neib_budget = fixed_neib_nodes
        self.max_nodes[-1] -= self.neib_budget
        # if self.neib_budget < len(self.agent_subset):
        #     self.neib_budget = 0
        # print(f"total amount for this batch:{total_amount}")
        # print(f"resources in subset:{np.array([self.max_nodes[agent.num] for agent in neib_agents])} for agents:{np.array([agent.num for agent in neib_agents])}")
        return self.neib_budget

class neib_fixed_150(neib_resources):
    neib_budget_name = "neighbourhood-fixed-150"

    def neib_pool(self, neib_agents: [AgentAlg], fixed_neib_nodes: int = 150,
                  cp_graph: dict = [str, List[AgentAlg]], prefix: int = 5):
        self.agent_subset = neib_agents
        if self.max_nodes[-1] < fixed_neib_nodes:
            fixed_neib_nodes = self.max_nodes[-1]
        self.neib_budget = fixed_neib_nodes
        self.max_nodes[-1] -= self.neib_budget
        # if self.neib_budget < len(self.agent_subset):
        #     self.neib_budget = 0
        # print(f"total amount for this batch:{total_amount}")
        # print(f"resources in subset:{np.array([self.max_nodes[agent.num] for agent in neib_agents])} for agents:{np.array([agent.num for agent in neib_agents])}")
        return self.neib_budget

class neib_fixed_prefix(neib_resources):
    neib_budget_name = "neighbourhood-fixed-prefix"
    len_cp_grap = 0
    fixed_amount = 0
    def neib_pool(self, neib_agents: [AgentAlg], fixed_neib_nodes: int = 150,
                  cp_graph: dict = [str, List[AgentAlg]], prefix: int =5):
        self.agent_subset = neib_agents
        try:
            #self.fixed_amount = self.max_nodes[-1]/(len(self.cp_graph))
            self.fixed_amount = prefix*16
            self.fixed_amount = tuple(self.fixed_amount)
        except:
            pass
        if self.max_nodes[-1] < self.fixed_amount:
            self.neib_budget = self.max_nodes[-1]
        else:
            self.neib_budget = self.fixed_amount
        self.max_nodes[-1] -= self.neib_budget
        # if self.neib_budget < len(self.agent_subset):
        #     self.neib_budget = 0
        # print(f"total amount for this batch:{total_amount}")
        # print(f"resources in subset:{np.array([self.max_nodes[agent.num] for agent in neib_agents])} for agents:{np.array([agent.num for agent in neib_agents])}")
        return self.neib_budget

class neib_shared(neib_resources):
    # neib_budget_name = "neighbourhood-shared"
    neib_budget_name = "Shared"
    def neib_pool(self, neib_agents:[AgentAlg], fixed_neib_nodes: int=50, cp_graph: dict=[str, List[AgentAlg]],prefix: int = 5):
        self.agent_subset = neib_agents
        self.neib_budget = self.max_nodes[-1]
        self.max_nodes[-1] -= self.neib_budget
        #print(f"total amount for this batch:{total_amount}")
        #print(f"resources in subset:{np.array([self.max_nodes[agent.num] for agent in neib_agents])} for agents:{np.array([agent.num for agent in neib_agents])}")
        # print(f"this batch proportions: {np.array([self.proportions[agent.num] for agent in self.agent_subset])}")
        # print(f"total neighborhood budget {self.neib_budget} | left in bank: {self.max_nodes[-1]}")
        return self.neib_budget

class neib_agents_distribution:
    neib_agents_name: str
    def agent_distribution(self):
        pass
    def neib_resource_subtraction(self,neib_agents:[AgentAlg],index):
        pass
    def return_resources(self):
        pass

class neib_agents_shared(neib_agents_distribution):
    neib_agents_name = "agents-shared"
    def agent_distribution(self):
        total_sum = 0
        for agent in self.agent_subset:
            self.max_nodes[agent.num] = int(self.neib_budget/len(self.agent_subset))
            total_sum += self.max_nodes[agent.num]
        self.neib_budget -= total_sum
        if total_sum <= 0:
            self.neib_budget = 0
        return np.array([self.max_nodes[agent.num] for agent in self.agent_subset])
    def neib_resource_subtraction(self, agent_num):
        index_list = []
        #print(f"resources left: {self.neib_budget} | {self.max_nodes[-1]}")
        for agent in self.agent_subset:
            index_list.append(agent.num)
        for index in index_list:
            if self.max_nodes[index] > 0:
                self.max_nodes[index] -= 1
                #print(f"resources in subset:{np.array([self.max_nodes[agent.num] for agent in self.agent_subset])} for agents:{np.array([agent.num for agent in self.agent_subset])}")
                return True
        return False
    def return_resources(self):
       # print(
            #f"Returning: resources in subset:{np.array([self.max_nodes[agent.num] for agent in self.agent_subset])} for agents:{np.array([agent.num for agent in self.agent_subset])}")
        for agent in self.agent_subset:
            self.neib_budget += self.max_nodes[agent.num]
        for agent in self.agent_subset:
            try:
                self.proportions[agent.num] = self.neib_budget/len(self.agent_subset)
            except TypeError:
                pass
            self.max_nodes[agent.num] = 0
        self.max_nodes[-1] += self.neib_budget
        self.neib_budget = 0
        #print(f"remain:{self.max_nodes[-1]}")
        return self.max_nodes[-1]

class neib_agents_evenly_split(neib_agents_distribution):
    neib_agents_name = "agents-evenly-split"
    def agent_distribution(self):
        total_sum = 0
        for agent in self.agent_subset:
            self.max_nodes[agent.num] = int(self.neib_budget/len(self.agent_subset))
            total_sum += self.max_nodes[agent.num]
        self.neib_budget -= total_sum
        if total_sum <= 0:
            self.neib_budget = 0
        #print(
        #    f"proportions in subset:{np.array([self.proportions[agent.num] for agent in self.agent_subset])} for agents:{np.array([agent.num for agent in self.agent_subset])}")
        #print(f"total amount for this batch:{total_amount}")
        #print(f"resources in subset:{np.array([self.max_nodes[agent.num] for agent in self.agent_subset])} for agents:{np.array([agent.num for agent in self.agent_subset])}")
        return np.array([self.max_nodes[agent.num] for agent in self.agent_subset])
    def neib_resource_subtraction(self, agent_num):
        index_list = []
        if self.max_nodes[agent_num] > 0:
            self.max_nodes[agent_num] -= 1
            #print(f"resources in subset:{np.array([self.max_nodes[agent.num] for agent in self.agent_subset])} for agents:{np.array([agent.num for agent in self.agent_subset])}")
        #    print(f"left in agent:{self.max_nodes[agent_num]} | left in bank: {self.max_nodes[-1]}")
            return True
        return False
    def return_resources(self):
        total_amount = 0
        #print(
            #f"Returning: resources in subset:{np.array([self.max_nodes[agent.num] for agent in self.agent_subset])} for agents:{np.array([agent.num for agent in self.agent_subset])}")
        for agent in self.agent_subset:
            self.neib_budget += self.max_nodes[agent.num]
        for agent in self.agent_subset:
            try:
                self.proportions[agent.num] = int(self.neib_budget/len(self.agent_subset))
            except TypeError:
                pass
            self.max_nodes[agent.num] = 0
        self.max_nodes[-1] += self.neib_budget
        self.neib_budget =0
        #if self.max_nodes[-1] < len(self.agent_subset):
        #print(f"remain:{self.max_nodes[-1]}")
        return self.max_nodes[-1]

class neib_agents_portion(neib_agents_distribution):
    neib_agents_name = "agent-propotion"
    def agent_distribution(self):
        total_sum = 0
        sum_proportions = 0
        for agent in self.agent_subset:
            sum_proportions += self.proportions[agent.num]
        for agent in self.agent_subset:
            self.max_nodes[agent.num] = int(self.neib_budget*self.proportions[agent.num]/sum_proportions)
            total_sum += self.max_nodes[agent.num]
        self.neib_budget -= total_sum
        if total_sum <= 0:
            self.neib_budget = 0
        #print(f"total amount for this batch:{total_amount}")
        #print(f"resources in subset:{np.array([self.max_nodes[agent.num] for agent in neib_agents])} for agents:{np.array([agent.num for agent in neib_agents])}")
        return np.array([self.max_nodes[agent.num] for agent in self.agents_subset])
    def neib_resource_subtraction(self, agent_num):
        index_list = []
        if self.max_nodes[agent_num] > 0:
            self.max_nodes[agent_num] -= 1
            #print(f"resources in subset:{np.array([self.max_nodes[agent.num] for agent in self.agent_subset])} for agents:{np.array([agent.num for agent in self.agent_subset])}")
            return True
        return False
    def return_resources(self):
        total_amount = 0
        #print(
        #    f"Returning: resources in subset:{np.array([self.max_nodes[agent.num] for agent in self.agent_subset])} for agents:{np.array([agent.num for agent in self.agent_subset])}")
        for agent in self.agent_subset:
            total_amount += self.max_nodes[agent.num]
            if self.neib_budget_name != "neighbourhood-proportions":
                try:
                    self.proportions[agent.num] = self.max_nodes[agent.num]
                except TypeError:
                    pass
        for agent in self.agent_subset:
            self.max_nodes[agent.num] = 0
        self.max_nodes[-1] += total_amount
        #print(f"remain:{self.max_nodes[-1]}")
        return self.max_nodes[-1]

class resource_and_neib_distributions:
    @staticmethod
    def create(distribution_class, neighbor_class=None, neighbor_agents_class=None, resources_per_agent=0, num_agents=0):
        #mixins = tuple([neighbor_class] if neighbor_class else [])
        #DynamicClass = create_resource_class(distribution_class, *mixins)
        DynamicClass = create_resource_class(distribution_class, neighbor_class, neighbor_agents_class)
        return DynamicClass(resources_per_agent=resources_per_agent, num_agents=num_agents)