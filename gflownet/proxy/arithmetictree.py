import torch

from typing import List
from torchtyping import TensorType

from gflownet.proxy.base import Proxy

from gflownet.envs.arithmetictree import ArithmeticTree

class ArithmeticScorer(Proxy):
    def __init__(
        self, 
        completion_reward: float,
        leaf_in_stock_reward: float,
        leaf_not_in_stock_reward: float,
        operation_count_reward: float,
        base_reward = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.completion_reward = completion_reward
        self.leaf_in_stock_reward = leaf_in_stock_reward
        self.leaf_not_in_stock_reward = leaf_not_in_stock_reward
        self.operation_count_reward = operation_count_reward
        self.base_reward = base_reward

    def setup(self, env = None):
        if env:
            # self.stock = env.stock
            self.max_n_nodes = env.max_n_nodes
            if self.base_reward == None:
                self.base_reward = 0.0
                if self.completion_reward < 0.0:
                    self.base_reward += -self.completion_reward
                if self.leaf_in_stock_reward < 0.0:
                    self.base_reward += -self.leaf_in_stock_reward * (self.max_n_nodes - 1.0)
                if self.leaf_not_in_stock_reward < 0.0:
                    self.base_reward += -self.leaf_not_in_stock_reward * (self.max_n_nodes - 1.0)
                if self.operation_count_reward < 0.0:
                    self.base_reward += -self.operation_count_reward * (env.max_operations)
            self.num_operation_classes = len(env.operations)

    def score_state(self, state: ArithmeticTree) -> float:
        is_complete = state.is_complete()
        leaf_in_stock_count = 0
        leaf_not_in_stock_count = 0
        for leaf_idx in state.get_leaf_nodes():
                if state[leaf_idx]["in_stock"]:
                    leaf_in_stock_count += 1
                else:
                    leaf_not_in_stock_count += 1
        operation_count = state.n_operations
        score = sum([
            self.base_reward,
            self.completion_reward * is_complete,
            self.leaf_in_stock_reward * leaf_in_stock_count,
            self.leaf_not_in_stock_reward * leaf_not_in_stock_count,
            self.operation_count_reward * operation_count,
        ])
        return score

    def __call__(
        self, states: List[ArithmeticTree]
    ) -> TensorType["batch"]:
        scores = torch.tensor(
            list(map(self.score_state, states)), 
            device=self.device
        )
        return -1.0 * scores