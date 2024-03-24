import torch

from typing import List
from torchtyping import TensorType

from gflownet.proxy.base import Proxy

class ReactionTreeScorer(Proxy):
    def __init__(
        self, 
        completetion_reward: float,
        leaf_in_stock_reward: float,
        leaf_not_in_stock_reward: float,
        reaction_count_reward: float,
        base_reward = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.completetion_reward = completetion_reward
        self.leaf_in_stock_reward = leaf_in_stock_reward
        self.leaf_not_in_stock_reward = leaf_not_in_stock_reward
        self.reaction_count_reward = reaction_count_reward
        self.base_reward = base_reward

    def setup(self, env = None):
        if env:
            # self.stock = env.stock
            self.max_n_nodes = env.max_n_nodes
            if self.base_reward == None:
                self.base_reward = 0.0
                if self.completetion_reward < 0.0:
                    self.base_reward += -self.completetion_reward
                if self.leaf_in_stock_reward < 0.0:
                    self.base_reward += -self.leaf_in_stock_reward * (self.max_n_nodes - 1.0)
                if self.leaf_not_in_stock_reward < 0.0:
                    self.base_reward += -self.leaf_not_in_stock_reward * (self.max_n_nodes - 1.0)
                if self.reaction_count_reward < 0.0:
                    self.base_reward += -self.reaction_count_reward * (env.max_reactions)

    def __call__(
        self, states: TensorType["batch", "state_dim"]
    ) -> TensorType["batch"]:
        leaf = states[:, :, -1] == 0
        in_stock = states[:, :, -2] == 1
        in_stock_count = in_stock.sum(axis=-1)
        not_in_stock = states[:, :, -2] == 0
        leaf_not_in_stock = leaf & not_in_stock
        leaf_not_in_stock_count = leaf_not_in_stock.sum(axis=-1)
        leaf_in_stock = leaf & in_stock
        leaf_in_stock_count = leaf_in_stock.sum(axis=-1)
        reaction_count = (states[:, :, -3] != -1).sum(axis=-1)
        incomplete = 1.0 * leaf_not_in_stock.any(axis=-1)
        is_complete = 1.0 - incomplete
        score = sum([
            self.base_reward,
            self.completetion_reward * is_complete,
            self.leaf_in_stock_reward * leaf_in_stock_count,
            self.leaf_not_in_stock_reward * leaf_not_in_stock_count,
            self.reaction_count_reward * reaction_count,
        ])
        return -1.0 * score