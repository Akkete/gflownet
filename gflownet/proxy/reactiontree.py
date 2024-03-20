import torch

from typing import List
from torchtyping import TensorType

from gflownet.proxy.base import Proxy

class ReactionTreeScorer(Proxy):
    def __init__(self, normalize: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.normalize = normalize

    def setup(self, env = None):
        if env:
            # self.stock = env.stock
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
        # # dummy test reward 1: more molecules in stock is better
        # in_stock = states[:, :, -2].sum(axis=-1)
        # return -1.0 * in_stock / self.norm
        # # test reward 2
        # leaf = states[:, :, -1] == 0
        # not_in_stock = states[:, :, -2] == 0
        # leaf_not_in_stock = (leaf & not_in_stock).sum(axis=-1)
        # reactions = (states[:, :, -3] != -1).sum(axis=-1)
        # return (- 21.0 * 3 # self.max_n_nodes 
        #         + 20.0 * leaf_not_in_stock 
        #         + 1.0 * reactions)
        # test reward 3: binary reward
        leaf = states[:, :, -1] == 0
        not_in_stock = states[:, :, -2] == 0
        leaf_not_in_stock = (leaf & not_in_stock).any(axis=-1)
        return 1.0 * leaf_not_in_stock - 1.0
