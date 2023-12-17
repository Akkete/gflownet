from __future__ import annotations
from typing import List, Optional, Tuple

from gflownet.envs.base import GFlowNetEnv

def flat_map(f, xs):
    return [y for ys in xs for y in f(ys)]

class ArithmeticNode:

    def get_leaves(self) -> List[ArithmeticNode]:
        if self.children:
            return flat_map(lambda x: x.get_leaves(), self.children)
        else:
            return [self]

class NumberNode(ArithmeticNode):

    def __init__(self, value: int, operation: Optional[OperationNode] = None):
        self.value = value
        self.operation = operation
        if operation:
            self.children = [self.operation]
        else:
            self.children = []
    
    def expand(self, op: str, a: int, b: int):
        self.operation = OperationNode(op, NumberNode(a), NumberNode(b))
        self.children = [self.operation]

class OperationNode(ArithmeticNode):

    def __init__(self, op: str, a: NumberNode, b: NumberNode):
        self.op = op
        self.a = a
        self.b = b
        self.children = [self.a, self.b]

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
        max_depth: int = 10, 
        min_int: int = -100, 
        max_int: int = +100,
        operations: List[str] = ['+', '*'],
        stock: List[int] = [1, -1, 2, -2, 3, -3],
        target: int = 0,
    ):
        self.max_depth = max_depth
        self.min_int = min_int
        self.max_int = max_int
        self.stock = stock
        # End-of-sequence action
        self.eos = (0, 0, -1)
        # The initial state is a tree with just the target
        self.source = NumberNode(target)
        # List of integers in leaves, initially contains just the target
        self.leaf_nodes = [target]

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
                    elif op == '*':
                        a = x / b
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
            state = self.state.clone().detach()
        if done is None:
            done = self.done
        if done:
            return [True for _ in range(self.policy_output_dim)]
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
            x, b, op = action
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
            self.leaf_nodes.append(leaf_to_expand.get_leaves())
            # Return
            return self.state, action, True

        


    


    

    