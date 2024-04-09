from __future__ import annotations
from functools import cache
from typing import List, Optional, Tuple, Union, Dict, Set
from torchtyping import TensorType
from numpy.typing import NDArray

import networkx as nx

import numpy as np

import torch
from torch.nn.functional import one_hot, pad

import copy

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import set_device

class ArithmeticTree:
    """
    Reperesents the parse tree of an arithmetic expression.

    The arithmetic tree is instantinated by supplying the target and stock.
    It should then be modified with the methods
    - `select_leaf(idx)`, 
    - `unselect_leaf()`,
    - `expand(operation, operands)`, and
    - `unexpand(idx)`.
    """
    def __init__(self, target: int, stock: List[int]):
        self.graph = nx.DiGraph()
        self.graph.add_node(0, integer=target, operation=-1, in_stock=0)
        self.stock = stock
        self._selected_leaf: Optional[int] = None
        self.n_operations = 0
    
    def copy(self) -> ArithmeticTree:
        """
        Returns a copy of the object. 
        
        The graph object is not shared between copies, but stock is.
        """
        copy_of_self = copy.copy(self)
        copy_of_self.graph = copy_of_self.graph.copy()
        return copy_of_self
    
    def __repr__(self) -> str:
        return str(list(self.graph.nodes))

    def __len__(self) -> int:
        return len(self.graph)
    
    def __getitem__(self, idx) -> Dict[str, int]:
        return self.graph.nodes[idx]
    
    def children(self, idx: int) -> List[int]:
        # TODO: Unnecessary?
        return list(self.graph.successors(idx))
    
    def get_leaf_nodes(self) -> List[int]:
        # TODO: Unused?
        return [n for n, d in self.graph.out_degree() if d==0]
    
    def get_parent(self, idx: int) -> Optional[int]:
        """Given an index, return parent index. For roote"""
        # TODO: Unused?
        # Assuming tree struucture, there should be at most one predecessor
        return next(self.graph.predecessors(idx), None)
    
    def is_complete(self) -> bool:
        leaf_indices = self.get_leaf_nodes()
        return all(self.graph.nodes[i]["in_stock"] for i in leaf_indices)

    def get_next_idx(self) -> int:
        return max(self.graph.nodes) + 1

    def get_selected_leaf(self) -> Optional[int]:
        return self._selected_leaf

    def select_leaf(self, idx: int) -> None:
        """Sets the selected leaf."""
        assert  self.graph.out_degree(idx) == 0 , "Must select a leaf."
        self._selected_leaf = idx

    def unselect_leaf(self) -> None:
        """Sets selected leaf to None. Needed as backward action."""
        self._selected_leaf = None

    def expand(self, operation: int, operands: List[int]) -> None:
        """
        Expand selected leaf with operation opid and a list of operands.
        Unsets selected leaf.
        """
        assert self._selected_leaf != None, "A leaf must be selected to expand."
        self.graph.nodes[self._selected_leaf]["operation"] = operation
        for operand in operands:
            new_idx = self.get_next_idx()
            self.graph.add_node(
                new_idx,
                integer=operand,
                operation=-1,
                in_stock=1 * (operand in self.stock)
            )
            self.graph.add_edge(self._selected_leaf, new_idx)
        self._selected_leaf = None
        self.n_operations += 1

    def unexpand(self, idx: int) -> None:
        """
        Reverse the expansion of a node. Deletes child nodes of the node at idx
        and unsets the operator attribute at idx. The unexpanded node becomes
        the selected leaf. Needed as backward action.
        """
        children = self.children(idx)
        assert (
            self[idx]["operation"] != -1 and
            all([d == 0 for _, d in self.graph.out_degree(children)])
            ), "To unexpand a node, all of its children must be leaves."
        self.graph.remove_nodes_from(children)
        self.graph.nodes[idx]["operation"] = -1
        self._selected_leaf = idx
        self.n_operations -= 1
    
    def get_unexpandable(self) -> List[int]:
        """Returns a list of nodes that could be uexpanded."""
        result: List[int] = []
        for node in self.graph.nodes:
            children = self.children(node)
            if (
                self[node]["operation"] != -1 and
                all([d == 0 for _, d in self.graph.out_degree(children)])
            ):
                result.append(node)
        return result

class ArithmeticBuilder(GFlowNetEnv):
    """
    Environment that generates valid arithmetic calculations.

    The goal is to create an arithmetic calculation that produces a target
    integer starting from only integers in 'stock'. A state is a bipartite tree
    of integers and arithmetic operations. In each step one leaf integer `x` is
    expanded into two integers `a` and `b` by applying an operation `op` to it 
    such that `x = a op b`.
    """

    def __init__(
        self, 
        min_int: int = -9, 
        max_int: int = +9,
        operations: List[str] = ['+', '*'],
        stock: List[int] = [1, -1, 2, -2],
        target: int = 0,
        max_operations: int = 10, 
        allow_early_eos: bool = True,
        **kwargs,
    ):
        assert target <= max_int, "Target can't be larger than max_int."
        assert target >= min_int, "Target can't be smaller than min_int."
        assert operations, "Operations can't be empty."
        assert (
            all(op in ['+', '*'] for op in operations)
            ), "The only supported operations are '+' and '*'."
        # Pytorch device may be supplied as a kwarg
        self.device = set_device(kwargs["device"])
        # Maximum number of opeartions
        # TODO: optionally use maximum depth instead
        self.max_operations = max_operations
        # The range of possible ints must be limited
        self.min_int = min_int
        self.max_int = max_int
        self.int_range = range(self.min_int, self.max_int + 1)
        # What index should be used to represent a missing integer
        self.no_int = min_int - 1
        # Operations, srock
        self.operations = operations
        self.stock = stock # stored in state class
        # Is early stopping allowed
        self.allow_early_eos = allow_early_eos
        # End-of-sequence action
        self.eos = (0, 0, 0)
        # The initial state is a tree with just the target
        self.source = ArithmeticTree(target, stock)
        # Max number of nodes is two times the number of opeartions plus one
        # This comes from the assumption that each operation takes in
        # two operands.
        self.max_n_nodes = self.max_operations * 2 + 1
        # Base class init
        super().__init__(**kwargs)

    @cache
    def operation_mask(self, op: str, x: int) -> NDArray[np.bool]:
        mask: NDArray[np.bool] = np.full(len(self.int_range), True)
        for i, b in enumerate(self.int_range):
            if op == '+':
                a = x - b
            elif op == '*' and b != 0:
                a = x / b
            else:
                a = None
            if a in self.int_range:
                mask[i] = False
        return mask

    def get_action_space(self) -> List[Tuple[int, int, int]]:
        """
        Constructs a list with all possible actions, including eos.

        There are two types of actions, one for choosing which leaf to expand
        and another for choosing how to expand that leaf.

        Actions are represented by tuples
        - The end-of-sequence (eos) action looks like this: (0, 0, 0)
        - Leaf-selection actions look like this: `(1, idx, 0)`
        - Expansion actions look like this: `(2, opid, b)`

        Let `x` be the integer at `idx`, `op` be the operation at `opid`, 
        and `inv` be inverse of the operation. Then, the child values are 
        `a` and `b`, such that `x = a op b`. Only `b` needs to be supplied to
        the action, since `a` can be calculated as `a = x inv b`.

        The first `self.max_n_nodes` actions are leaf-selection actions.
        After that, the next `len(self.int_range)` actions are the expansion 
        actions for `opid` 0, then again the same number of actions for the
        next opid and so on. The last action is the end-of-sequence action.
        """
        actions: List[Tuple[int, int, int]] = []
        for idx in range(self.max_n_nodes):
            actions.append((1, idx, 0))
        for opid in range(len(self.operations)):
            for b in self.int_range:
                    actions.append((2, opid, b))
        actions.append(self.eos)
        return actions

    def get_mask_invalid_actions_forward(
        self, 
        state: Optional[ArithmeticTree] = None, 
        done: Optional[bool] = None,
    ):
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [True for _ in range(self.policy_output_dim)]
        if state.n_operations >= self.max_operations:
            return [True for _ in range(self.policy_output_dim - 1)] + [False]
        
        # Start out with everything masked
        mask = [True for _ in range(self.policy_output_dim)]

        # Unmask eos action if early stopping is allowed or all leaf are 
        # in stock.
        if (self.allow_early_eos or state.is_complete()):
            mask[-1] = False

        # If no leaf is selected we just unmask leaf indices that are not in 
        # stock and we are done
        selected_leaf = state.get_selected_leaf()
        if selected_leaf == None:
            for leaf_idx in state.get_leaf_nodes():
                if not state[leaf_idx]["in_stock"]:
                    mask[leaf_idx] = False
            return mask
        
        # If a leaf x is selected, we check which (op, b) pairs can be applied 
        # to it so that a = x inv b is in self.int_range.
        x = state[selected_leaf]["integer"] 
        for opid, op in enumerate(self.operations):
            relevant_slice = slice(
                self.max_n_nodes + len(self.int_range) * opid,
                self.max_n_nodes + len(self.int_range) * (opid + 1)
            )
            operation_mask = self.operation_mask(op, x)
            mask[relevant_slice] = operation_mask

        return mask

    def step(
        self, action: Tuple[int, int, int], skip_mask_check: bool = False
    ) -> Tuple[List[int], Tuple[int, int, int], bool]:
        """
        Executes step given an action.

        Args
        ----
        action : tuple
            Action to be executed. 
            An action is a tuple int values `(x, b, opid)`

        skip_mask_check : bool
            If True, skip computing forward mask of invalid actions to check if 
            the action is valid.

        Returns
        -------
        self.state : list
            The sequence after executing the action

        action : tuple
            Action executed

        valid : bool
            False, if the action is not allowed for the current state.
        """
        do_step, self.state, action = self._pre_step(action, skip_mask_check)
        if not do_step:
            return self.state, action, False
        # If action is eos
        if action == self.eos:
            self.done = True
            self.n_actions += 1
            return self.state, self.eos, True
        # If action is normal action, perform action
        action_type, idx_or_opid, b = action
        # Action type 1: leaf selection
        if action_type == 1:
            assert (
                self.state.get_selected_leaf() == None
                ), "Leaf already slected."
            idx = idx_or_opid
            try:
                self.state.select_leaf(idx)
                self.n_actions += 1
                return self.state, action, True
            except AssertionError:
                return self.state, action, False
        # Action type 2: expansion
        if action_type == 2:
            opid = idx_or_opid
            op = self.operations[opid]
            x = self.state[self.state.get_selected_leaf()]["integer"]
            if x == None:
                return self.state, action, False
            if op == '+':
                a = x - b
            elif op == '*':
                if x == 0 and b == 0:
                    a = 0
                elif b == 0:
                    return self.state, action, False
                else:
                    a = x / b
            else:
                raise Exception("Operator must be '+' or '*'.")
            # If a is not in the range of allowed integers, 
            # the action is invalid
            if a not in self.int_range:
                return self.state, action, False
            try:
                self.state.expand(operation=opid, operands=[int(a), b])
            # Increment number of actions, update state and return
                self.n_actions += 1
                return self.state, action, True
            except AssertionError:
                return self.state, action, False
        # If the excution gets here something went wrong
        raise Exception("There is a bug if the control reaches here.")
        return self.state, action, False

    def get_parents(
        self,
        state: Optional[ArithmeticTree] = None,
        done: Optional[bool] = None,
        action: Optional[Tuple] = None,
    ) -> Tuple[List[ArithmeticTree], List[Tuple[int, int, int]]]:
        """
        Determines all parents and actions that lead to state.

        In continuous environments, `get_parents()` should return only the 
        parent from which action leads to state.

        Args
        ----
        state : list
            Representation of a state

        done : bool
            Whether the trajectory is done. 
            If None, done is taken from instance.

        action : None
            Ignored

        Returns
        -------
        parents : list
            List of parents in state format

        actions : list
            List of actions that lead to state for each parent in parents
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [state], [self.eos,]

        # If leaf is selected, the previous action was selecting that leaf.
        selected_leaf = state.get_selected_leaf()
        if selected_leaf != None:
            parent = state.copy()
            parent.unselect_leaf()
            return [parent,], [(1, selected_leaf, 0)]

        # If there was no selected leaf, the previous action was an expansion.
        parents: List[ArithmeticTree] = []
        actions: List[Tuple[int, int, int]] = []
        unexpandable = state.get_unexpandable()
        for idx in unexpandable:
            b = state[list(state.children(idx))[-1]]["integer"]
            actions.append((2, state[idx]["operation"], b))
            parent = state.copy()
            parent.unexpand(idx)
            parents.append(parent)
        return parents, actions

    def ints2one_hot(self, integers: int) -> TensorType["one_hot_length"]:
        one_hot_length = len(self.int_range)
        integers_shifted = torch.tensor(integers) - self.min_int
        ints_one_hot = one_hot(integers_shifted, num_classes=one_hot_length)
        return ints_one_hot

    def state2tensor(
        self, 
        state: ArithmeticTree
    ) -> TensorType["max_n_nodes", "one_hot_length+len(self.operations)+3"]:
        ints = [value for _, value in state.graph.nodes(data="integer")]
        ints_one_hot = self.ints2one_hot(ints)
        ops = [value for _, value in state.graph.nodes(data="operation")]
        # We transform opids to one hot vectors
        # The +1 is so that -1 goes to 0
        ops_one_hot = one_hot(torch.tensor(ops) + 1, len(self.operations) + 1)
        in_stock = [value for _, value in state.graph.nodes(data="in_stock")]
        in_stock_tensor = torch.tensor(in_stock)
        leaf_nodes = torch.tensor(state.get_leaf_nodes())
        is_leaf = one_hot(leaf_nodes, num_classes=self.max_n_nodes).sum(axis=0)
        active_leaf = state.get_selected_leaf()
        if active_leaf == None:
            is_active = torch.zeros(self.max_n_nodes)
        else:
            is_active = one_hot(torch.tensor(active_leaf), self.max_n_nodes)
        padding = (0, self.max_n_nodes - in_stock_tensor.shape[0])
        ints_padded = pad(ints_one_hot, (0, 0) + padding, value=0)
        ops_padded = pad(ops_one_hot, (0, 0) + padding, value=0)
        in_stock_padded = pad(in_stock_tensor, padding, value=0).unsqueeze(1)
        is_leaf_unsqueezed = is_leaf.unsqueeze(1)
        is_active_unsqueezed = is_active.unsqueeze(1)
        result_tensor = torch.cat(
            (
                ints_padded, 
                ops_padded, 
                in_stock_padded, 
                is_leaf_unsqueezed, 
                is_active_unsqueezed
            ), axis=-1).to(device=self.device)
        return result_tensor

    def states2proxy(
        self, states: List[ArithmeticTree]
    ) -> List[ArithmeticTree]:
        return states

    def states2oracle(
        self, states: List[ArithmeticTree]
    ) -> List[ArithmeticTree]:
        return states

    def states2policy(
        self, states: List[ArithmeticTree]
    ) -> TensorType["batch_size", "policy_input_dim"]:
        return torch.stack(list(map(self.state2tensor, states)), axis = 0).flatten(1, -1)

    def state2readable(self, state: Optional[ArithmeticTree] = None):
        """
        Converts a state into a readable arithmetic expression.
        """
        if state is None:
            state = self.state.copy()
        def depth_first_traversal(idx: int) -> str:
            children = state.children(idx)
            if children:
            # node has children
                lc = depth_first_traversal(children[0])
                rc = depth_first_traversal(children[-1])
                op = self.operations[state[idx]["operation"]]
                return f"({lc}{op}{rc})"
            else:
            # node has no children
                value = state[idx]["integer"]
                if value < 0:
                    return f"({value})"
                else:
                    return f"{value}"
        return depth_first_traversal(0)

    def reset(self, env_id: Union[int, str] = None):
        """
        Resets the environment.

        Args
        ----
        env_id: int or str
            Unique (ideally) identifier of the environment instance, 
            used to identify the trajectory generated with this environment. 
            If None, uuid.uuid4() is used.

        Returns
        -------
        self
        """
        # Most of the resetting is handled by the base class reset method
        super().reset(env_id)
        return self