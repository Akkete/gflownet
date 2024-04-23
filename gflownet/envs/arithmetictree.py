from __future__ import annotations
from functools import cache
from typing import List, Optional, Tuple, Union, Dict
from typing_extensions import Self
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

    The arithmetic tree is instantinated by supplying the stock and a list of
    possible target integers.
    It starts as an empty graph.
    It should then be modified with the methods
    - `select_target(target_value)`,
    - `unselect_target()`,
    - `select_leaf(idx)`, 
    - `unselect_leaf()`,
    - `expand(operation, operands)`, and
    - `unexpand(idx)`.
    """
    def __init__(self, stock: List[int], targets: List[int]):
        self.graph = nx.DiGraph()
        self.stock = stock
        self.targets = targets
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
        """Given an index, return parent index. For root"""
        # TODO: Unused?
        # Assuming tree struucture, there should be at most one predecessor
        return next(self.graph.predecessors(idx), None)
    
    def is_complete(self) -> bool:
        if len(self.graph.nodes) == 0:
            return False
        leaf_indices = self.get_leaf_nodes()
        return all(self.graph.nodes[i]["in_stock"] for i in leaf_indices)

    def get_next_idx(self) -> int:
        return max(self.graph.nodes) + 1

    def get_selected_leaf(self) -> Optional[int]:
        return self._selected_leaf

    def select_target(self, target_value: int) -> Self:
        """
        Sets the target. Should only be called when the tree is empty.
        The arget should be from the target list.
        """
        assert len(self.graph.nodes) == 0, "Graph should be empty."
        assert target_value in self.targets, "Invalid target."
        self.graph.add_node(0, integer=target_value, operation=-1, in_stock=0)
        return self

    def unselect_target(self) -> Self:
        """
        Unselects target returning to the empty state. Should only be called 
        when there is one node left in the tree and it has index 0. Needed as a
        backward action.
        """
        assert list(self.graph.nodes) == [0], "Only root should be in graph."
        self.graph.remove_node(0)
        return self

    def select_leaf(self, idx: int) -> Self:
        """Sets the selected leaf."""
        assert  self.graph.out_degree(idx) == 0 , "Must select a leaf."
        self._selected_leaf = idx
        return self

    def unselect_leaf(self) -> Self:
        """Sets selected leaf to None. Needed as backward action."""
        self._selected_leaf = None
        return self

    def expand(self, operation: int, operands: List[int]) -> Self:
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
        return self

    def unexpand(self, idx: int) -> Self:
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
        return self
    
    def get_unexpandable(self) -> List[int]:
        """Returns a list of nodes that could be unexpanded."""
        result: List[int] = []
        for node in self.graph.nodes:
            children = self.children(node)
            if (
                self[node]["operation"] != -1 and
                all([d == 0 for _, d in self.graph.out_degree(children)])
            ):
                result.append(node)
        return result

class ActionType:
    STOP = 0
    SELECT_TARGET = 1
    SELECT_LEAF = 2
    EXPAND = 3

class ArithmeticBuilder(GFlowNetEnv):
    """
    Environment that generates valid arithmetic expressions.

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
        targets: List[int] = [9, -9, 8, -8, 7, -7, 0],
        max_operations: int = 10, 
        allow_early_eos: bool = True,
        **kwargs,
    ):
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
        # Operations, stock, targets
        self.operations = operations
        self.stock = stock # also stored in state class
        self.targets = targets # also stored in state class
        # Is early stopping allowed
        self.allow_early_eos = allow_early_eos
        # End-of-sequence action
        self.eos = (ActionType.STOP, 0, 0)
        # The initial state is an empty arithmetic tree object
        self.source = ArithmeticTree(stock, targets)
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

        There are four types of actions. The main ones are one for choosing 
        which leaf to expand and another for choosing how to expand that leaf.
        The action types for selecting a target and stop are only used once.

        Actions are represented by tuples
        - The end-of-sequence (eos) action looks like this: (0, 0, 0)
        - Target-selection actions look like this: `(1, target_value, 0)`
        - Leaf-selection actions look like this: `(2, idx, 0)`
        - Expansion actions look like this: `(3, opid, b)`

        Let `x` be the integer at `idx`, `op` be the operation at `opid`, 
        and `inv` be inverse of the operation. Then, the child values are 
        `a` and `b`, such that `x = a op b`. Only `b` needs to be supplied to
        the action, since `a` can be calculated as `a = x inv b`.

        The first `len(self.targets)` actions are target-selection actions.
        The next `self.max_n_nodes` actions are leaf-selection actions.
        After that, the next `len(self.int_range)` actions are the expansion 
        actions for `opid` 0, then again the same number of actions for the
        next opid and so on. The last action is the end-of-sequence action.
        """
        actions: List[Tuple[int, int, int]] = []
        for target_value in self.targets:
            actions.append((ActionType.SELECT_TARGET, target_value, 0))
        for idx in range(self.max_n_nodes):
            actions.append((ActionType.SELECT_LEAF, idx, 0))
        for opid in range(len(self.operations)):
            for b in self.int_range:
                actions.append((ActionType.EXPAND, opid, b))
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

        # Unmask eos action if early stopping is allowed or if the state is
        # complete.
        if (self.allow_early_eos or state.is_complete()):
            mask[-1] = False

        # Starting indices of each action type
        target_selection_start = 0
        leaf_selection_start = target_selection_start + len(self.targets)
        expansion_start = leaf_selection_start + self.max_n_nodes

        # If target is not yet selected, unmask target selection actions
        # and we are done
        if len(state.graph.nodes) == 0:
            slice_ = slice(target_selection_start, leaf_selection_start)
            mask[slice_] = (False for _ in range(target_selection_start, 
                                                 leaf_selection_start))
            return mask

        selected_leaf = state.get_selected_leaf()
        # If no leaf is selected we just unmask leaf indices that are not in 
        # stock and we are done
        if selected_leaf == None:
            for leaf_idx in state.get_leaf_nodes():
                if not state[leaf_idx]["in_stock"]:
                    mask[leaf_selection_start + leaf_idx] = False
            return mask
        
        # If a leaf x is selected, we check which (op, b) pairs can be applied 
        # to it so that a = x inv b is in self.int_range.
        x = state[selected_leaf]["integer"] 
        for opid, op in enumerate(self.operations):
            relevant_slice = slice(
                expansion_start + len(self.int_range) * opid,
                expansion_start + len(self.int_range) * (opid + 1)
            )
            operation_mask = self.operation_mask(op, x)
            mask[relevant_slice] = operation_mask

        # If all other actions are invalid, make eos valid
        if all(mask):
            mask[-1] = False

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
        action_type, action_value, b = action
        # Action type 1: target selection
        if action_type == ActionType.SELECT_TARGET:
            assert len(self.state.graph.nodes) == 0, "Target already selected."
            target_value = action_value
            self.state.select_target(target_value)
            self.n_actions += 1
            return self.state, action, True
        # Action type 2: leaf selection
        if action_type == ActionType.SELECT_LEAF:
            assert (
                self.state.get_selected_leaf() == None
                ), "Leaf already slected."
            idx = action_value
            try:
                self.state.select_leaf(idx)
                self.n_actions += 1
                return self.state, action, True
            except AssertionError:
                return self.state, action, False
        # Action type 3: expansion
        if action_type == ActionType.EXPAND:
            opid = action_value
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
            return [parent,], [(ActionType.SELECT_LEAF, selected_leaf, 0)]

        # If there is precisely one node in the tree, the previous action was 
        # selecting the target.
        if len(state.graph.nodes) == 1:
            target_value = state.graph.nodes[0]["integer"]
            parent = state.copy()
            parent.unselect_target()
            return [parent,], [(ActionType.SELECT_TARGET, target_value, 0)]

        # If there was no selected leaf, the previous action was an expansion.
        parents: List[ArithmeticTree] = []
        actions: List[Tuple[int, int, int]] = []
        unexpandable = state.get_unexpandable()
        for idx in unexpandable:
            b = state[list(state.children(idx))[-1]]["integer"]
            actions.append((ActionType.EXPAND, state[idx]["operation"], b))
            parent = state.copy()
            parent.unexpand(idx)
            parents.append(parent)
        return parents, actions

    def ints2one_hot(
        self, 
        integers: List[int]
    ) -> TensorType["len(integers)", "one_hot_length"]:
        one_hot_length = len(self.int_range)
        if len(integers) == 0:
            return torch.empty((0, one_hot_length))
        integers_shifted = torch.tensor(integers) - self.min_int
        ints_one_hot = one_hot(integers_shifted, num_classes=one_hot_length)
        return ints_one_hot

    def state2tensor(
        self, 
        state: ArithmeticTree
    ) -> TensorType["max_n_nodes+1", "one_hot_length+len(self.operations)+4"]:
        ints = [value for _, value in state.graph.nodes(data="integer")]
        ints_one_hot = self.ints2one_hot(ints)
        ops = [value for _, value in state.graph.nodes(data="operation")]
        # We transform opids to one hot vectors
        # The +1 is so that -1 goes to 0
        if ops:
            ops_one_hot = one_hot(torch.tensor(ops) + 1, len(self.operations) + 1)
        else:
            ops_one_hot = torch.empty((0, len(self.operations) + 1))
        in_stock = [value for _, value in state.graph.nodes(data="in_stock")]
        in_stock_tensor = torch.tensor(in_stock)
        leaf_nodes = torch.tensor(state.get_leaf_nodes())
        if leaf_nodes.shape[0] != 0:
            is_leaf = one_hot(leaf_nodes, num_classes=self.max_n_nodes).sum(axis=0)
        else:
            is_leaf = torch.zeros(self.max_n_nodes)
        active_leaf = state.get_selected_leaf()
        if active_leaf == None:
            is_active = torch.zeros(self.max_n_nodes)
            active_leaf_tensor = torch.zeros(
                len(self.int_range) + len(self.operations) + 4)
        else:
            is_active = one_hot(torch.tensor(active_leaf), self.max_n_nodes)
            active_leaf_one_hot = self.ints2one_hot([state[active_leaf]["integer"]])
            active_leaf_tensor = torch.cat(
                (
                    active_leaf_one_hot.squeeze(), 
                    torch.zeros(len(self.operations) + 1),
                    torch.tensor([0, 1, 1])
                ))
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
                is_active_unsqueezed,
            ), axis=-1).to(device=self.device)
        result_tensor_with_active_appended = torch.cat(
            (result_tensor, active_leaf_tensor.unsqueeze(0)), axis=0
        )
        return result_tensor_with_active_appended
    
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
            # node does not exist
            if idx not in state.graph.nodes:
                return "."
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
        expression = depth_first_traversal(0)
        if state.graph.nodes[0]:
            target = str(state.graph.nodes[0]["integer"])
        else:
            target = "."
        return expression + "=" + target

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