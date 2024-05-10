import torch
from torch import nn
import random
import numpy as np
import copy
import math

class Node():
    def __init__(self,
                 name: str,
                 day: int,
                 previous_state: list,
                 children_maximum: int = 5,
                 parent = None,
                 asset_memory: list = None,
                 date_memory: list = None,
                 state_memory: list = None,
                 rewards_memory: list = None,
                 actions_memory: list = None,
                 rollout_length: int = None, # 从该结点往后的rollout长度
                 last_weights: np.ndarray = None,
                 portfolio_value: float = None,
                 portfolio_value_vector: list = None,
                 cost: float = None,
                 trades: int = None,
                 portfolio_return_memory: list = None,
                 turbulence: float = None,
                 x_prev: np.ndarray = None,
                 ) -> None:
        self.name = name
        self.day = day
        self.children_maximum = children_maximum
        self.children = []
        self.is_terminal = False   # 用来标识该结点是否是树可扩展的最深结点。可以使用day和最大day的差距来决定。
        self.value = 0
        self.parent = parent
        self.rollout_length = rollout_length
        self.asset_memory = copy.deepcopy(asset_memory)
        self.date_memory = copy.deepcopy(date_memory)
        self.rewards_memory = copy.deepcopy(rewards_memory)
        self.actions_memory = copy.deepcopy(actions_memory)
        self.cost = cost
        self.trades = trades

        self.state_memory = copy.deepcopy(state_memory)
        self.previous_state = copy.deepcopy(previous_state)
        self.turbulence = turbulence

        self.last_weights = copy.deepcopy(last_weights)
        self.portfolio_value = portfolio_value
        self.portfolio_value_vector = copy.deepcopy(portfolio_value_vector)
        self.portfolio_return_memory = copy.deepcopy(portfolio_return_memory)

        self.visit = 0

        self.x_prev = x_prev

    def is_fully_expanded(self):
        if len(self.children) >= self.children_maximum:
            return True
        else:
            return False
    

class MCTS():
    def __init__(self,
                 mcts_gamma,
                 expand_length,
                 C=0.5,
                 random_select_prob=0.5
                 ) -> None:
        self.root = None
        self.gamma = mcts_gamma
        self.expand_length = expand_length
        self.node_num = 0
        self.C = C
        self.random_select_prob = random_select_prob # 取0即不执行mcts

    
    def search(self):
        node = self.root
        layer = 0
        while True:  
            node.visit += 1  # 遍历次数+1
            if len(node.children) == 0:
                # 无孩子，扩展
                break
            elif random.uniform(0, 1) < self.random_select_prob or node.is_fully_expanded():   
                # 条件1：随机结果符合
                # 条件2：节点达到最大拓展数
                node = self.select(node)
                layer+=1
            else:
                # 扩展
                break
        return node,layer


    def select(self, node):
        # 这个过程DDPG不进行训练。
        scores = []
        for n,child in enumerate(node.children):
            score = child.value / max(child.visit,1) + self.C * math.sqrt(float(child.parent.visit)) / float(1 + child.visit)
            scores.append(score)
        probs = np.array(scores)
        probs = probs / np.sum(probs)
        idx = list(np.random.multinomial(1, probs)).index(1)
        best_node = node.children[idx]
        return best_node
    
    def back_prop(self,node,returns):
        cur = node
        decay_rate = self.gamma
        while cur.parent is not None:
            cur.value += decay_rate*returns
            cur = cur.parent
            decay_rate *= decay_rate
    
    def cal_mcts_reward(self,asset):
        return asset / 1e6
    