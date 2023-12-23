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
        self.operations = operations
        self.stock = stock
        # End-of-sequence action
        self.eos = (0, 0, -1)
        # The initial state is a tree with just the target
        self.source = NumberNode(target)
        # 
        self.statetorch2oracle = self.state2oracle
        # Base class init
        super().__init__(**kwargs)

    def number_node_to_tensor(self, node: NumberNode) -> TensorType:
        """
        Converts an integer number into a one-hot vector representation.
        """
        length = self.max_int - self.min_int + 1
        tensor = torch.zeros(length)
        tensor[node.value - self.min_int] = 1.0
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
        state: Optional = None, 
        done: Optional[bool] = None,
    ):
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [True for _ in range(self.policy_output_dim)]
        if self.n_actions >= self.max_operations:
            return [True for _ in range(self.policy_output_dim-1)] + [False]
        mask = [False for _ in range(self.policy_output_dim)]
        for idx, action in enumerate(self.action_space[:-1]):
            x, _, _ = action
            if x not in map(lambda x: x.value, self.leaf_nodes):
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
                lambda leaf: leaf.value == x, 
                self.leaf_nodes
            ))
            leaf_to_expand.expand(op, a, b)
            # Update leaf nodes
            self.leaf_nodes.remove(leaf_to_expand)
            self.leaf_nodes = leaf_to_expand.get_leaves()
            # Increment number of actions and return
            self.n_actions += 1
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
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [state], [(self.eos,)]
        parents = []
        actions = []
        root = state.copy()
        queue = [(root, state)]
        while queue:
            node = queue.pop()
            if node.operation and any(map(lambda x: x.operation, node.operation.children)):
                [(root.copy(), x) for x in node.operation.children]
            elif node.operation:
                pass

        return parents, actions

    def state2policy(
        self, state: Optional[TensorType["height", "width"]] = None
    ) -> TensorType["height", "width"]:
        """
        Prepares a state in "GFlowNet format" for the policy model.

        See: state2oracle()
        """
        return self.state2oracle(state).flatten()

    def state2oracle(
        self, state: Optional = None
    ) -> TensorType["height", "width"]:
        """
        Prepares a state in "GFlowNet format" for the oracles.
        """
        if state is None:
            state = self.state.copy()
        return sum(map(self.number_node_to_tensor, self.leaf_nodes))

    def statebatch2policy(
        self, states: List[List]
    ) -> TensorType["batch_size", "policy_input_dim"]:
        """
        Converts a list of states into a format suitable for a machine learning 
        models.
        """
        return torch.stack(list(map(self.state2oracle, states)), axis = 0)

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
        # List of leaf nodes is maintained, starts as source
        self.leaf_nodes = [self.state]
        return self

    

