from __future__ import annotations
from typing import List, Optional, Tuple, Union
from torchtyping import TensorType

import numpy as np
import pandas as pd

import torch
from torch.nn.functional import pad

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import set_device

from aizynthfinder.context.stock import Stock
from aizynthfinder.chem import Molecule
from rdkit.Chem.AllChem import ReactionFromSmarts
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import MolToSmiles

from pathlib import Path
import copy

PROJECT_ROOT = Path(__file__).parents[2]

# Load stock
stock_file = PROJECT_ROOT / "data/reactiontree/zinc_stock.hdf5"
STOCK = Stock()
STOCK.load(str(stock_file), "zinc")
STOCK.select("zinc")

class ReactionTree:
    """
    Represents a reaction tree.
    """
    def __init__(
        self,
        molecules: List[str],
        reactions: List[int], # reaction index or -1 if no reaction
        in_stock: List[bool],
        children: List[List[int]] # empty list signifies no children
    ):
        self.__len = len(molecules)
        assert (self.__len == len(reactions) and
                self.__len == len(in_stock) and
                self.__len == len(children)), "Lengths should match"
        self.molecules = molecules
        self.reactions = reactions
        self.in_stock = in_stock
        self.children = children

    def copy(self):
        return copy.deepcopy(self)
    
    def __len__(self):
        return self.__len
    
    def __getitem__(self, idx):
        return {"molecule": self.molecules[idx],
                "reaction": self.reactions[idx],
                "in_stock": self.in_stock[idx],
                "children": self.children[idx]}

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
        # TODO: optionally use maximum depth instead
        self.max_reactions = max_reactions
        # If template_file is given as a relative path, 
        # it is interpreted to be relative to the project root.
        template_path = Path(template_file)
        if not template_path.is_absolute():
            template_path = PROJECT_ROOT / template_path
        # Load templates
        if ".csv" in template_path.suffixes:
            self.templates: pd.DataFrame = pd.read_csv(
                str(template_path), index_col=0, sep="\t"
            )
        else:
            self.templates = pd.read_hdf(str(template_path), "table")
        self.reactions = self.templates["retro_template"].apply(ReactionFromSmarts)
        # Allow or not early termination
        self.allow_early_eos = allow_early_eos
        # End-of-sequence action
        self.eos = (-1,)
        # The initial state is a tree with just the target
        self.source = ReactionTree(
            molecules = [target_smiles],
            reactions = [-1],
            in_stock = [False],
            children = [[]]
        )
        # The maximum number of nodes in the reaction tree 
        # is five times number of reactions plus one, 
        # because each reaction has a maximum of five children
        self.max_n_nodes = max_reactions *  5 + 1
        # Base class init
        super().__init__(**kwargs)

    def get_leaf_indices(
        self, 
        state: Optional[ReactionTree] = None
    ) -> List[int]:
        if state == None:
            state = self.state.copy()
        def depth_first_traversal(idx: int) -> List[int]:
            # node at index has no children (is leaf)
            if state.children[idx] == []:
               return [idx]
            # node has children
            else:
                list_of_results = [depth_first_traversal(child) 
                                   for child in state.children[idx]]
                flattened_result = sum(list_of_results, [])
                return flattened_result
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
            # node at index has no children (is leaf)
            if state.children[idx] == []:
                # check if it is in stock or not
                if state.in_stock[idx] == False:
                    return idx
                else:
                    return None
            # node has children
            else:
                for child in state.children[idx]:
                    result_from_child = depth_first_traversal(child)
                    if result_from_child:
                        return result_from_child
                # if no child produces a reuslt, return None
                return None
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
        # reaction = ReactionFromSmarts(
        #     self.templates["retro_template"][reaction_id])
        reaction = self.reactions[reaction_id]
        molecule = MolFromSmiles(state.molecules[idx])
        reactants = reaction.RunReactants([molecule])
        if len(reactants) > 0:
            reactants = reactants[0]
        else:
            return None # warning bad solution
        state.reactions[idx] = reaction_id
        for reactant in reactants:
            smiles = MolToSmiles(reactant)
            aizynth = Molecule(rd_mol = reactant)
            state.molecules.append(smiles)
            state.reactions.append(-1)
            try:
                in_stock = aizynth in STOCK
                state.in_stock.append(in_stock)
            except:
                return None
            state.children[idx].append(len(state.molecules)-1)
            state.children.append([])
            # Check for bad molecules
            if MolFromSmiles(smiles) == None:
                return None
        return state

    def node_to_tensor(self, idx: int) -> TensorType["fingerprint_length"]:
        """
        Converts a molecule into a vector fingerprint.
        """
        try: 
            molecule = self.state.molecules[idx]
        except:
            molecule = None
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
                reaction = self.reactions[reaction_index]
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
        for idx, child_list in enumerate(state.children):
            if child_list and child_list[-1] == len(state.molecules)-1:
                parent_state = state.copy()
                parent_state.reactions[idx] = -1
                parent_state.children[idx] = []
                for _ in child_list:
                    del parent_state.molecules[-1]
                    del parent_state.reactions[-1]
                    del parent_state.in_stock[-1]
                    del parent_state.children[-1]
                action = (state.reactions[idx],)
                return [parent_state], [action]

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
        # Fingerprints are provided by a helper function 
        # that also deals with missing values.
        # No padding needed.
        fingerprints = [self.node_to_tensor(i) for i in range(self.max_n_nodes)]
        fngrprnt_tensor = torch.stack(fingerprints, axis = 0)
        # The other lists are first created as variable length and then padded.
        reaction_tensor = torch.tensor(state.reactions, 
                                       device=self.device)
        in_stock_tensor = torch.tensor(state.in_stock, 
                                       device=self.device)
        children_tensor = torch.tensor(list(map(len, state.children)), 
                                           device=self.device)
        # Pad and reshape
        padding = (0, self.max_n_nodes - reaction_tensor.shape[0])
        reaction_tensor = pad(reaction_tensor, padding)
        in_stock_tensor = pad(in_stock_tensor, padding)
        children_tensor = pad(children_tensor, padding)
        reaction_tensor = reaction_tensor.unsqueeze(dim=1)
        in_stock_tensor = in_stock_tensor.unsqueeze(dim=1)
        children_tensor = children_tensor.unsqueeze(dim=1)
        # Return the concatenated tensor
        return torch.cat(
            (
                fngrprnt_tensor, 
                reaction_tensor, 
                in_stock_tensor,
                children_tensor,
            ), 
            axis = -1).to(
            dtype = torch.float32,
            device = self.device
        )

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

    def state2readable(self, state: Optional[ReactionTree] = None):
        """
        Converts a state into a readable summary.
        """
        if state is None:
            state = self.state.copy()
        reactions = list(filter(lambda x: x != -1, state.reactions))
        leaf_mols_in_stock = []
        leaf_mols_not_in_stock = []
        for i in range(len(state)):
            if len(state[i]["children"]) == 0:
                if state[i]["in_stock"]:
                    leaf_mols_in_stock.append(state[i]["molecule"])
                else:
                    leaf_mols_not_in_stock.append(state[i]["molecule"])
        leaf_mols_in_stock_str = ", ".join(leaf_mols_in_stock)
        leaf_mols_not_in_stock_str = ", ".join(leaf_mols_not_in_stock)
        reactions_str = ", ".join(map(str, reactions))
        return (f"Number of reactions: {len(reactions)}"
                f"\nReaction indices: {reactions_str}"
                f"\nMolecules in stock ({len(leaf_mols_in_stock)}): {leaf_mols_in_stock_str}"
                f"\nMissing from stock ({len(leaf_mols_not_in_stock)}): {leaf_mols_not_in_stock_str}")
    
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