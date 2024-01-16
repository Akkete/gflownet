import torch

from typing import List
from torchtyping import TensorType

from gflownet.proxy.base import Proxy

class ArithmeticScorer(Proxy):
    def __init__(self, normalize: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.normalize = normalize

    def setup(self, env = None):
        if env:
            self.stock = env.stock
            self.max_n_nodes = env.max_n_nodes

    @property
    def norm(self):
        if self.normalize:
            return 1.0 * self.max_n_nodes
        else:
            return 1.0

    def __call__(
        self, states: TensorType["batch", "state_dim"]
    ) -> TensorType["batch"]:
        # dummy test reward 1: more numbers in stock is better
        in_stock = sum(states[:, :, 0]==i for i in self.stock).bool()
        return in_stock.sum(axis=1) / self.norm
        # dummy test reward 2: more bigger numbers is better
        # return -1.0 * states.sum(axis=(1, 2))