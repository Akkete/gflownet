from __future__ import annotations
from typing import List, Optional, Tuple, Union
from torchtyping import TensorType

import numpy as np
import pandas as pd

import torch

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import set_device

from aizynthfinder.context.stock import Stock
from aizynthfinder.chem import Molecule
from rdkit.Chem.AllChem import ReactionFromSmarts
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import MolToSmiles

import os
import copy

# Load stock
stock_file = ("/m/home/home9/94/anttona2/data/Documents/research_project"
             "/gflownet/data/reactiontree/zinc_stock.hdf5")
STOCK = Stock()
STOCK.load(stock_file, "zinc")
STOCK.select("zinc")

class ReactionTree:
    """
    Represents a reaction tree. 
    """
    def __init__(
        self,
        molecules: List[Union[str, None]],
        reactions: List[int], # reaction index or -1 if no reaction
        in_stock: List[bool]
    ):
        self.molecules = molecules
        self.reactions = reactions
        self.in_stock = in_stock

    def copy(self):
        return copy.deepcopy(self)

class ReactionTreeBuilder(GFlowNetEnv):
    """
    Environment that generates retrosynthesis routes.
    """

    def __init__(
        self, 
        template_file: str,
        # stock_file: str,
        target_smiles: str,
        max_reactions: int = 5, 
        allow_early_eos: bool = False,
        **kwargs,
    ):
        # Pytorch device may be supplied as a kwarg
        self.device = set_device(kwargs["device"])
        # Maximum number of reactions
        # TODO: use maximum depth instead
        self.max_reactions = max_reactions
        # Load templates
        if template_file.endswith(".csv.gz") or template_file.endswith(".csv"):
            self.templates: pd.DataFrame = pd.read_csv(
                template_file, index_col=0, sep="\t"
            )
        else:
            self.templates = pd.read_hdf(template_file, "table")
        # Set target
        # self.target = Molecule(smiles = target_smiles)
        # Allow or not early termination
        self.allow_early_eos = allow_early_eos
        # End-of-sequence action
        self.eos = (-1,)
        # The initial state is a tree with just the target
        self.max_n_nodes = 2**(max_reactions + 1) - 1
        self.source = ReactionTree(
            molecules = [None for _ in range(self.max_n_nodes)],
            reactions = [-1 for _ in range(self.max_n_nodes)],
            in_stock = [False for _ in range(self.max_n_nodes)]
        )
        self.source.molecules[0] = target_smiles
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
        parent = ReactionTreeBuilder._get_parent(k)
        if parent is None:
            return None
        left = ReactionTreeBuilder._get_left_child(parent)
        right = ReactionTreeBuilder._get_right_child(parent)
        return left if k == right else right

    def get_leaf_indices(
        self, 
        state: Optional[ReactionTree] = None
    ) -> List[int]:
        if state == None:
            state = self.state.copy()
        def depth_first_traversal(idx: int) -> List[int]:
            # index is out of bounds
            if idx >= self.max_n_nodes:
                return []
            # no node at index
            elif state.molecules[idx] == None:
                return []
            # node at index has no children (is leaf)
            elif idx >= self.max_n_nodes / 2 or state.reactions[idx] == -1:
               return [idx]
            # node has children
            else:
                lc = depth_first_traversal(self._get_left_child(idx))
                rc = depth_first_traversal(self._get_right_child(idx))
                return lc + rc
        return depth_first_traversal(0)
    
    def get_active_leaf(
        self,
        state: Optional[TensorType] = None
    ) -> int:
        """
        Find first leaf that is not in stock.
        """
        if state == None:
            state = self.state.copy()
        def depth_first_traversal(idx: int) -> Optional[int]:
            # index is out of bounds
            if idx >= self.max_n_nodes:
                return None
            # no node at index
            elif state.molecules[idx] == None:
                return None
            # node at index has no children (is leaf)
            elif idx >= self.max_n_nodes / 2 or state.reactions[idx] == -1:
                # check if it is in stock or not
                if state.in_stock[idx] == False:
                    return idx
                else:
                    return None
            # node has children
            else:
                lc = depth_first_traversal(self._get_left_child(idx))
                if lc:
                    return lc
                else:
                    rc = depth_first_traversal(self._get_right_child(idx))
                    return rc
        return depth_first_traversal(0)

    def expand(
        self,
        idx: int, 
        reaction_id: int,
        state: Optional[TensorType] = None
    ) -> TensorType:
        """
        Expand node at index idx with reaction reaction_id.
        Return updated state.
        """
        if state:
            state = state.copy()
        else:
            state = self.state.copy()
        assert state.molecules[idx] != None, "Trying to expand an empty node."
        reaction = ReactionFromSmarts(
            self.templates["retro_template"][reaction_id])
        molecule = MolFromSmiles(state.molecules[idx])
        reactants = reaction.RunReactants([molecule])
        if len(reactants) > 0:
            reactants = reactants[0]
        else:
            return None # warning bad solution
        first_reactant = reactants[0]
        first_smiles = MolToSmiles(first_reactant)
        first_aizynth = Molecule(rd_mol = first_reactant)
        state.reactions[idx] = reaction_id
        lc_idx = self._get_left_child(idx)
        state.molecules[lc_idx] = first_smiles
        # Checking the stock sometimes fails because of unkekulizable molecules
        # This is an ugly way to guard against that
        try:
            state.in_stock[lc_idx] = first_aizynth in STOCK
        except:
            return None
        # Check for bad molecules
        if MolFromSmiles(first_smiles) == None:
            return None
        if len(reactants) > 1:
            second_reactant = reactants[1]
            second_smiles = MolToSmiles(second_reactant)
            second_aizynth = Molecule(rd_mol = second_reactant)
            rc_idx = self._get_right_child(idx)
            state.molecules[rc_idx] = second_smiles
            try:
                state.in_stock[rc_idx] = second_aizynth in STOCK
            except:
                return None
            # Check for bad molecules
            if MolFromSmiles(second_smiles) == None:
                return None
        return state

    def node_to_tensor(self, idx: int) -> TensorType["fingerprint_length"]:
        """
        Converts a molecule into a vector fingerprint.
        """
        molecule = self.state.molecules[idx]
        if molecule:
            aizynthfinder_mol = Molecule(smiles = molecule)
            tensor = torch.tensor(
                aizynthfinder_mol.fingerprint(2, 64), device = self.device)
        else:
            tensor = torch.zeros((64,), device = self.device)
        return tensor

    def get_action_space(self):
        """
        Constructs a list with all possible actions, including eos.

        Every retroreaction template in the templates list is one action,
        referred by its template code (index).
        """
        actions = [(x,) for x in self.templates.index]
        actions.append(self.eos)
        return actions

    def get_mask_invalid_actions_forward(
        self, 
        state: Optional[TensorType] = None, 
        done: Optional[bool] = None,
    ):
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [True for _ in range(self.policy_output_dim)]
        if self.n_actions >= self.max_reactions:
            return [True for _ in range(self.policy_output_dim - 1)] + [False]

        # Each action is checked separately
        mask = [False for _ in range(self.policy_output_dim)]
        active_leaf = self.get_active_leaf(state)
        if active_leaf == None:
            return [True for _ in range(self.policy_output_dim - 1)] + [False]
        molecule = MolFromSmiles(state.molecules[active_leaf])
        for idx, action in enumerate(self.action_space[:-1]):
                reaction_index = action[0]
                reaction = ReactionFromSmarts(
                    self.templates["retro_template"][reaction_index])
                mask[idx] = len(reaction.RunReactants([molecule])) == 0
        if not self.allow_early_eos and not all(mask[:-1]):
            mask[-1] = True
        return mask

    def step(
        self, action: Tuple[int], skip_mask_check: bool = False
    ) -> Tuple[List[int], Tuple[int], bool]:
        """
        Executes step given an action.

        Args
        ----
        action : tuple
            Action to be executed. 
            An action is an index referring to the list of templates.

        skip_mask_check : bool
            If True, skip computing forward mask of invalid actions to check if 
            the action is valid.

        Returns
        -------
        self.state : list
            The state after executing the action

        action : tuple
            Action executed

        valid : bool
            False, if the action is not allowed for the current state.
        """
        
        do_step, self.state, action = self._pre_step(action, skip_mask_check=True)
        if not do_step:
            return self.state, action, False
        # If action is eos
        if action == self.eos:
            self.done = True
            self.n_actions += 1
            return self.state, self.eos, True
        # If action is normal action, perform action
        reaction_id = action[0]
        active_leaf = self.get_active_leaf()
        if active_leaf == None:
            return self.state, action, False
        updated_state = self.expand(active_leaf, reaction_id, self.state)
        if updated_state == None:
            return self.state, action, False # warning bad solution
        # Update leaf nodes (should currently be unused, TODO: remove)
        self.leaf_indices.remove(active_leaf)
        self.leaf_indices.append(self._get_left_child(active_leaf))
        self.leaf_indices.append(self._get_right_child(active_leaf))
        # Increment number of actions, update state and return
        self.n_actions += 1
        self.state = updated_state
        return self.state, action, True

    def get_parents(
        self,
        state: Optional[ReactionTree] = None,
        done: Optional[bool] = None,
        action: Optional[Tuple] = None,
    ) -> Tuple[List, List]:
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
            return [state], [self.eos]
        parents = []
        actions = []
        stack = [0]
        while stack:
            idx = stack.pop()
            if state.reactions[idx] == -1:
                continue
            left_child = self._get_left_child(idx)
            right_child = self._get_right_child(idx)
            left_expanded = state.reactions[left_child] != -1
            right_expanded = state.reactions[right_child] != -1
            if left_expanded or right_expanded:
                stack.append(right_child)
                stack.append(left_child)
            else:
                parent_state = state.copy()
                parent_state.reactions[idx] = -1
                parent_state.molecules[right_child] = None
                parent_state.molecules[left_child] = None
                parent_state.in_stock[right_child] = False
                parent_state.in_stock[left_child] = False
                action = (state.reactions[idx],)
                parents.append(parent_state)
                actions.append(action)
        return parents[-1:], actions[-1:]

    def state2proxy(
        self, state: Optional[TensorType] = None
    ) -> TensorType["fingerprint_length"]:
        return self.state2oracle(state)

    def statebatch2proxy(
        self, states: List[TensorType]
    ) -> TensorType["batch", "fingerprint_length"]:
        return self.statebatch2oracle(states)

    def statetorch2proxy(
        self, states: TensorType["batch", "state_dim"]
    ) -> TensorType:
        return self.statetorch2oracle(states)

    def state2oracle(
        self, state: Optional[TensorType] = None
    ) -> TensorType["self.max_n_nodes", "fingerprint_length + 2"]:
        if state is None:
            state = self.state.copy()
        fingerprints = [self.node_to_tensor(i) for i in range(self.max_n_nodes)]
        fp_tensor = torch.stack(fingerprints, axis = 0)
        reaction_tensor = torch.tensor(state.reactions, 
                                       device = self.device).unsqueeze(dim = 1)
        in_stock_tensor = torch.tensor(state.in_stock, 
                                       device = self.device).unsqueeze(dim = 1)
        return torch.cat((fp_tensor, reaction_tensor, in_stock_tensor), 
                         axis = -1).to(
                         dtype = torch.float32,
                         device = self.device)

    def statebatch2oracle(
        self, states: List[TensorType]
    ) -> TensorType["batch", "state_oracle_dim"]:
        return torch.stack(list(map(self.state2oracle, states)), axis = 0)

    def statetorch2oracle(
        self, states: TensorType["", "batch"]
    ) -> TensorType["fingerprint_length", "batch"]:
        return self.statebatch2oracle(torch.unbind(states, dim=-1))

    def state2policy(
        self, state: Optional[TensorType] = None
    ) -> TensorType["fingerprint_length"]:
        if state is None:
            state = self.state.copy()
        active_leaf = self.get_active_leaf(state)
        if active_leaf:
            return self.node_to_tensor(active_leaf)
        else:
            return torch.zeros((64,), device = self.device)

    def statebatch2policy(
        self, states: List[List]
    ) -> TensorType["batch_size", "fingerprint_length"]:
        return torch.stack(
                list(map(self.state2policy, states)), 
                axis = 0
            ).flatten(start_dim=1)
    
    def statetorch2policy(self, states: TensorType) -> TensorType:
        return torch.stack(
                list(map(self.state2policy, torch.unbind(states, dim=-1))), 
                axis = 0
            ).flatten(start_dim=1)

    def policy2state(
        self, policy: Optional[TensorType["fingerprint_length"]] = None
    ) -> None:
        """
        Returns None to signal that the conversion is not reversible.
        """
        return None

    def state2readable(self, state: Optional[TensorType["state_dim"]] = None):
        """
        Converts a state into a readable list of leaf molecules.
        """
        if state is None:
            state = self.state.copy()
        leaf_molecules = [state.molecules[idx] for idx in self.get_leaf_indices(state)]
        return ", ".join(leaf_molecules)

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
        # List of leaf indices is maintained, starts as root
        self.leaf_indices = [0]
        return self