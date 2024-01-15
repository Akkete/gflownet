import torch

from typing import List
from torchtyping import TensorType

from gflownet.proxy.base import Proxy

class ArithmeticScorer(Proxy):
    def __init__(self, stock: List[int] = [1, -1, 2, -2, 3, -3], **kwargs):
        super().__init__(**kwargs)
        self.stock = stock

    def setup(self, env=None):
        if env:
            self.stock = env.stock

    def __call__(
        self, states: TensorType["batch", "state_dim"]
    ) -> TensorType["batch"]:
        # dummy test reward 1: more numbers in stock is better
        in_stock = sum(states[:, :, 0]==i for i in self.stock).bool()
        return in_stock.sum(axis=1)
        # dummy test reward 2: more bigger numbers is better
        # return states.sum(axis=(1, 2))