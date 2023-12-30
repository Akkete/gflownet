from __future__ import annotations
from typing import List, Optional, Tuple
import copy

from gflownet.envs.base import GFlowNetEnv
import torch

# utility functcion
def flat_map(f, xs):
    return [y for ys in xs for y in f(ys)]

class ArithmeticNode:

    def get_leaves(self) -> List[ArithmeticNode]:
        if self.children:
            return flat_map(lambda x: x.get_leaves(), self.children)
        else:
            return [self]

    def get_leaf_numbers(self) -> List[NumberNode]:
        leaves = self.get_leaves()
        for idx, leaf in enumerate(leaves):
            if isinstance(leaf, OperationNode):
                leaves[idx] = leaf.parent
        return leaves

    def get_leaf_operations(self) -> List[OperationNode]:
        leaves = self.get_leaves()
        leaf_operations = set()
        for leaf in leaves:
            if isinstance(leaf, OperationNode):
                leaf_operations.add(leaf)
            elif isinstance(leaf, NumberNode) and leaf.parent:
                leaf_operations.add(leaf)
        return list(leaf_operations)

    def copy(self) -> ArithmeticNode:
        return copy.deepcopy(self)
    

class NumberNode(ArithmeticNode):

    def __init__(
        self, 
        value: int, 
        operation: Optional[OperationNode] = None,
    ):
        self.value = value
        self.operation = operation
        if operation:
            self.children = [self.operation]
        else:
            self.children = []
        self.parent = None
    
    def expand(self, op: str, a: int, b: int):
        operation = OperationNode(op, NumberNode(a), NumberNode(b))
        operation.set_parent(self)
    
    def set_operation(self, operation: OperationNode):
        assert not self.operation, "Node already has a child operation."
        self.operation = operation
        self.children = [self.operation]
    
    def delete_operation(self):
        self.operation.parent = None
        self.operation = None
        self.children = None
        

class OperationNode(ArithmeticNode):

    def __init__(
        self, 
        op: str, 
        a: NumberNode, 
        b: NumberNode, 
    ):
        self.op = op
        self.a = a
        self.b = b
        self.children = [self.a, self.b]
        self.parent = None
    
    def set_parent(self, parent: NumberNode):
        assert not self.parent, "Node already has a parent."
        assert not parent.operation, "Parent already has a child operation."
        self.parent = parent
        parent.set_operation(self)
    
    def cut_from_parent(self):
        self.parent.delete_operation()


class ArithmeticBuilder(GFlowNetEnv):
    """
    Environment that generates valid arithmetic alculations.

    The goal is to create an arithmetic calculation that produces a target
    integer starting from only integers in 'stock'. A state is a bipartite tree
    of integers and arithmetic operations. In each step one leaf integer x is
    expanded into two integers a and b by applying an operation op to it such
    that x = a op b.
    """

    def __init__(
        self, 
        max_operations: int = 10, 
        min_int: int = -100, 
        max_int: int = +100,
        operations: List[str] = ['+', '*'],
        stock: List[int] = [1, -1, 2, -2, 3, -3],
        target: int = 0,
        **kwargs,
    ):
        self.max_operations = max_operations
        self.min_int = min_int
        self.max_int = max_int
        self.no_int = min_int - 1
        self.operations = operations
        self.stock = stock
        # End-of-sequence action
        self.eos = (0, 0, -1)
        # The initial state is a tree with just the target
        max_n_nodes = 2**max_operations - 1
        self.source = torch.stack((
            torch.full((max_n_nodes,), self.no_int),
            torch.full((max_n_nodes,), -1)
        ), axis = 1)
        self.source[0][0] = target
        # Base class init
        super().__init__(**kwargs)

    @staticmethod
    def _get_parent(k: int) -> Optional[int]:
        """
        Get node index of a parent of k-th node.
        """
        if k == 0:
            return None
        return (k - 1) // 2

    @staticmethod
    def _get_left_child(k: int) -> int:
        """
        Get node index of a left child of k-th node.
        """
        return 2 * k + 1

    @staticmethod
    def _get_right_child(k: int) -> int:
        """
        Get node index of a right child of k-th node.
        """
        return 2 * k + 2

    @staticmethod
    def _get_sibling(k: int) -> Optional[int]:
        """
        Get node index of the sibling of k-th node.
        """
        parent = Tree._get_parent(k)
        if parent is None:
            return None
        left = Tree._get_left_child(parent)
        right = Tree._get_right_child(parent)
        return left if k == right else right

    def _expand(
        self,
        idx: int, 
        opid: str, 
        a: int, 
        b: int, 
        state: Optional[TensorType] = None
    ) -> TensorType:
        """
        Expand node at index idx with operation opid and childs a and b.
        Return updated state.
        """
        if state:
            updated_state = state.clone().detach()
        else:
            updated_state = self.state.clone().detach()
        updated_state[idx][1] = opid
        updated_state[self._get_left_child(idx)][0] = a
        updated_state[self._get_right_child(idx)][0] = b
        return updated_state


    def node_to_tensor(self, idx: int) -> TensorType["one_hot_length"]:
        """
        Converts an integer number into a one-hot vector representation.
        """
        one_hot_length = self.max_int - self.min_int + 1
        tensor = torch.zeros(one_hot_length)
        tensor[int(self.state[idx][0] - self.min_int)] = 1.0
        return tensor

    def get_action_space(self):
        """
        Constructs a list with all possible actions, including eos.

        An action consists of choosing an operation op and integers a and b and
        attaching them to a leaf node x. To be valid the action should be 
        chosen such that x = a op b.

        Actions are represented by a tuple (x, b, opid). 
        The integer a can be calculated by a = x inv b.
        TODO: The integer x is automatically chosen for the agent.
        """
        actions = []
        for opid in range(len(self.operations)):
            op = self.operations[opid]
            for x in range(self.min_int, self.max_int):
                for b in range(self.min_int, self.max_int):
                    if op == '+':
                        a = x - b
                    elif op == '*' and b != 0:
                        a = x / b
                    else:
                        a = None
                    if a in range(self.min_int, self.max_int):
                        actions.append((x, b, opid))
        actions.append(self.eos)
        return actions

    def get_mask_invalid_actions_forward(
        self, 
        state: Optional[TensorType] = None, 
        done: Optional[bool] = None,
    ):
        if state is None:
            state = self.state.clone().detach()
        if done is None:
            done = self.done
        if done:
            return [True for _ in range(self.policy_output_dim)]
        if self.n_actions >= self.max_operations:
            return [True for _ in range(self.policy_output_dim - 1)] + [False]
        mask = [False for _ in range(self.policy_output_dim)]
        for idx, action in enumerate(self.action_space[:-1]):
            x, _, _ = action
            if x not in map(lambda idx: state[idx][0], self.leaf_indices):
                mask[idx] = True
        return mask

    def step(
        self, action: Tuple[int], skip_mask_check: bool = False
    ) -> Tuple[List[int], Tuple[int], bool]:
        """
        Executes step given an action.

        Args
        ----
        action : tuple
            Action to be executed. An action is a tuple int values (x, b, opid)

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
        else:
            x, b, opid = action
            op = self.operations[opid]
            if op == '+':
                a = x - b
            elif op == '*':
                a = x / b
            # Find leaf that matches x and expand it
            leaf_to_expand = next(filter(
                lambda idx: self.state[idx][0] == x, 
                self.leaf_indices
            ))
            updated_state = self._expand(leaf_to_expand, opid, a, b)
            # Update leaf nodes
            self.leaf_indices.remove(leaf_to_expand)
            self.leaf_indices.append(self._get_left_child(leaf_to_expand))
            self.leaf_indices.append(self._get_right_child(leaf_to_expand))
            # Increment number of actions, update state and return
            self.n_actions += 1
            self.state = updated_state
            return self.state, action, True

    def get_parents(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
        action: Optional[Tuple] = None,
    ) -> Tuple[List, List]:
        """
        Determines all parents and actions that lead to state.

        In continuous environments, get_parents() should return only the parent from
        which action leads to state.

        Args
        ----
        state : list
            Representation of a state

        done : bool
            Whether the trajectory is done. If None, done is taken from instance.

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
            state = self.state.clone().detach()
        if done is None:
            done = self.done
        if done:
            return [state], [self.eos,]
        parents = []
        actions = []
        stack = [0]
        while stack:
            idx = stack.pop()
            if state[idx][1] == -1:
                continue
            left_child = self._get_left_child(idx)
            right_child = self._get_right_child(idx)
            left_expanded = state[left_child][1] != -1
            right_expanded = state[right_child][1] != -1
            if left_expanded or right_expanded:
                stack.append(left_child)
                stack.append(right_child)
            else:
                parent_state = state.clone().detach()
                parent_state[idx][1] = -1
                parent_state[right_child] = self.no_int
                parent_state[left_child] = self.no_int
                action = (int(state[idx][0]), int(state[right_child][0]), int(state[idx][1]))
                parents.append(parent_state)
                actions.append(action)
        return parents, actions

    def state2proxy(
        self, state: Optional[TensorType] = None
    ) -> TensorType["one_hot_length"]:
        return self.state2oracle(state)

    def statebatch2proxy(
        self, states: List[TensorType]
    ) -> TensorType["batch", "one_hot_length"]:
        return self.statebatch2oracle(states)

    def statetorch2proxy(
        self, states: TorchType["batch", "state_dim"]
    ) -> TorchType:
        return self.statetorch2oracle(states)

    def state2policy(
        self, state: Optional[TensorType] = None
    ) -> TensorType["one_hot_length"]:
        return self.state2oracle(state)

    def state2oracle(
        self, state: Optional = None
    ) -> TensorType["one_hot_length"]:
        if state is None:
            state = self.state.clone().detach()
        return sum(map(self.node_to_tensor, self.leaf_indices))

    def statebatch2oracle(
        self, states: List[TensorType]
    ) -> TensorType["batch", "state_oracle_dim"]:
        return torch.stack(list(map(self.state2oracle, states)), axis = 0)

    def statetorch2oracle(
        self, states: TensorType["", "batch"]
    ) -> TensorType["one_hot_length", "batch"]:
        return self.statebatch2oracle(torch.unbind(states, dim=-1))

    def state2policy(self, state: Optional[TensorType] = None) -> TensorType:
        return self.state2oracle(state).flatten()

    def statebatch2policy(
        self, states: List[List]
    ) -> TensorType["batch_size", "policy_input_dim"]:
        return self.statebatch2oracle(states).flatten(start_dim=1)
    
    def statetorch2policy(self, states: TensorType) -> TensorType:
        return statetorch2oracle(states).flatten(start_dim=1)

    def policy2state(
        self, policy: Optional[TensorType["policy_input_dim"]] = None
    ) -> None:
        """
        Returns None to signal that the conversion is not reversible.
        """
        return None

    def reset(self, env_id: Union[int, str] = None):
        """
        Resets the environment.

        Args
        ----
        env_id: int or str
            Unique (ideally) identifier of the environment instance, used to identify
            the trajectory generated with this environment. If None, uuid.uuid4() is
            used.

        Returns
        -------
        self
        """
        # Most of the resetting is handled by the base class reset method
        super().reset(env_id)
        # List of leaf indices is maintained, starts as root
        self.leaf_indices = [0]
        return self

    

