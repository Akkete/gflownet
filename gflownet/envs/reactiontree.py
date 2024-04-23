from __future__ import annotations
from functools import lru_cache
from typing import List, Optional, Tuple, Union, Dict
from typing_extensions import Self
from torchtyping import TensorType
from numpy.typing import NDArray
import warnings

import numpy as np
import pandas as pd

import networkx as nx 

import torch
from torch.nn.functional import one_hot, pad

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import set_device

from aizynthfinder.context.stock import Stock, InMemoryInchiKeyQuery
from aizynthfinder.chem import Molecule
from rdkit.Chem.rdChemReactions import ReactionFromSmarts, ChemicalReaction
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import MolToSmiles

from pathlib import Path
import copy

PROJECT_ROOT = Path(__file__).parents[2]

# Stock and template list 
# Load stock
stock_file = PROJECT_ROOT / "external/reactiontree_data/zinc_stock.hdf5"
STOCK = Stock()
STOCK.load(InMemoryInchiKeyQuery(str(stock_file)), "zinc")
STOCK.select("zinc")

class ReactionTree:
    """
    Represents a retrosynthesis reaction tree.

    The tree contains the followinng fields for each node.
    - `molecule: str` -- smiles string of the molecule.
    - `reation: int` -- reaction index (referring to the the template list),
                        -1 is used to signify no rection.
    - `in_stock: bool` -- keeps track of which molecules are in stock
    Each node has an integer index. The indexes start from 0 for the root and
    after that are given out in the order that nodes are added.

    For instantiation, only the stock and a list of possible target molecules 
    is given. The reaction tree starts out as an empty graph and is then 
    modified with the following methods.
    - `select_target(tgt_idx)`,
    - `unselect_target()`,
    - `select_leaf(idx)`,
    - `unselect_leaf()`,
    - `expand(rxn_idx)`, and
    - `unexpand(idx)`.
    These methods correspond to the GFlowNet actions.
    """
    def __init__(
        self, 
        # stock: Stock, 
        templates: List[ChemicalReaction], 
        targets: List[str]
    ):
        self.graph = nx.DiGraph()
        # self.stock = stock
        self.templates: list[ChemicalReaction] = templates
        self.targets = targets
        self._selected_leaf: Optional[str] = None
        self.n_reactions = 0

    def copy(self):
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
        return (len(self.graph))
    
    def __getitem__(self, idx: int) -> Dict[str, int]:
        return self.graph.nodes[idx]
    
    def children(self, idx: int) -> List[int]:
        return list(self.graph.successors(idx))
    
    def get_leaf_nodes(self) -> List[int]:
        return [n for n, d in self.graph.out_degree() if d==0]
    
    def is_complete(self) -> bool:
        if len(self.graph.nodes) == 0:
            return False
        leaf_indices = self.get_leaf_nodes()
        return all(self.graph.nodes[i]["in_stock"] for i in leaf_indices)
    
    def get_next_idx(self) -> int:
        return max(self.graph.nodes) + 1
    
    def get_selected_leaf(self) -> Optional[int]:
        return self._selected_leaf
    
    def select_target(self, tgt_idx: int) -> Self:
        """
        Sets the target. Should only be called when the tree is empty.
        The arget should be from the target list.
        """
        assert len(self.graph.nodes) == 0, "Graph should be empty."
        assert tgt_idx in range(len(self.targets)), f"Invalid target {tgt_idx}."
        self.graph.add_node(
            0, 
            molecule=self.targets[tgt_idx], 
            reaction=-1, 
            in_stock=False
        )
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
    
    def expand(self, rxn_idx: int) -> Self:
        """
        Expand selected leaf with reaction rxn_id. Unsets selected leaf.
        """
        assert self._selected_leaf != None, "A leaf must be selected to expand."
        selected = self.get_selected_leaf()
        self.graph.nodes[selected]["reaction"] = rxn_idx
        reaction: ChemicalReaction = self.templates[rxn_idx]
        molecule: Molecule = MolFromSmiles(self[selected]["molecule"])
        reactants = reaction.RunReactants([molecule])
        assert len(reactants) > 0, "Reaction didn't produce reactants."
        reactants = reactants[0]
        for reactant in reactants:
            new_idx = self.get_next_idx()
            reactant_smiles = MolToSmiles(reactant)
            assert MolFromSmiles(reactant_smiles) != None, "Bad molecule"
            reactant_aizynth = Molecule(rd_mol=reactant)
            reactant_in_stock = reactant_aizynth in STOCK
            self.graph.add_node(
                new_idx,
                molecule=reactant_smiles,
                reaction=-1,
                in_stock=reactant_in_stock
            )
            self.graph.add_edge(selected, new_idx)
        self._selected_leaf = None
        self.n_reactions += 1
        return self
    
    def unexpand(self, idx: int) -> Self:
        """
        Reverse the expansion of a node. Deletes child nodes of the node at idx
        and unsets the reaction attribute at idx. The unexpanded node becomes
        the selected leaf. Needed as backward action.
        """
        children = self.children(idx)
        assert (
            self[idx]["reaction"] != -1 and
            all([d == 0 for _, d in self.graph.out_degree(children)])
            ), "To unexpand a node, all of its children must be leaves."
        self.graph.remove_nodes_from(children)
        self.graph.nodes[idx]["reaction"] = -1
        self._selected_leaf = idx
        self.n_reactions -= 1
        return self
    
    def get_unexpandable(self) -> List[int]:
        """Returns a list of nodes that could be unexpanded."""
        result: List[int] = []
        for node in self.graph.nodes:
            children = self.children(node)
            if (
                self[node]["reaction"] != -1 and
                all([d == 0 for _, d in self.graph.out_degree(children)])
            ):
                result.append(node)
        return result

class ActionType:
    STOP = 0
    SELECT_TARGET = 1
    SELECT_LEAF = 2
    EXPAND = 3

class ReactionTreeBuilder(GFlowNetEnv):
    """
    Environment that generates retrosynthesis routes.
    """

    def __init__(
        self, 
        template_file: str, # path
        # stock_file: str, # path
        target_file: str, # path
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
        self.reactions = list(self.templates["retro_template"].apply(ReactionFromSmarts))
        # Load target file
        target_path = Path(target_file)
        if not target_path.is_absolute():
            target_path = PROJECT_ROOT / target_path
        with open(target_path) as targets_file:
            self.targets = [line.rstrip() for line in targets_file]
        # Allow or not early termination
        self.allow_early_eos = allow_early_eos
        # End-of-sequence action
        self.eos = (ActionType.STOP, 0,)
        # The initial state is an empty reactiion tree object
        self.source = ReactionTree(self.reactions, self.targets)
        # The maximum number of nodes in the reaction tree 
        # is five times number of reactions plus one, 
        # because each reaction has a maximum of five children
        self.max_n_nodes = max_reactions *  5 + 1
        # Fingerprint length
        self.fingerprint_length = 2048
        # Base class init
        super().__init__(**kwargs)

    # def get_leaf_indices(
    #     self, 
    #     state: Optional[ReactionTree] = None
    # ) -> List[int]:
    #     if state == None:
    #         state = self.state.copy()
    #     def depth_first_traversal(idx: int) -> List[int]:
    #         # node at index has no children (is leaf)
    #         if state.children[idx] == []:
    #            return [idx]
    #         # node has children
    #         else:
    #             list_of_results = [depth_first_traversal(child) 
    #                                for child in state.children[idx]]
    #             flattened_result = sum(list_of_results, [])
    #             return flattened_result
    #     return depth_first_traversal(0)
    
    # def get_active_leaf(
    #     self,
    #     state: Optional[TensorType] = None
    # ) -> int:
    #     """
    #     Find first leaf that is not in stock.
    #     """
    #     if state == None:
    #         state = self.state.copy()
    #     def depth_first_traversal(idx: int) -> Optional[int]:
    #         # node at index has no children (is leaf)
    #         if state.children[idx] == []:
    #             # check if it is in stock or not
    #             if state.in_stock[idx] == False:
    #                 return idx
    #             else:
    #                 return None
    #         # node has children
    #         else:
    #             for child in state.children[idx]:
    #                 result_from_child = depth_first_traversal(child)
    #                 if result_from_child:
    #                     return result_from_child
    #             # if no child produces a reuslt, return None
    #             return None
    #     return depth_first_traversal(0)

    # def expand(
    #     self,
    #     idx: int, 
    #     reaction_id: int,
    #     state: Optional[TensorType] = None
    # ) -> TensorType:
    #     """
    #     Expand node at index idx with reaction reaction_id.
    #     Return updated state.
    #     """
    #     if state:
    #         state = state.copy()
    #     else:
    #         state = self.state.copy()
    #     assert state.molecules[idx] != None, "Trying to expand an empty node."
    #     # reaction = ReactionFromSmarts(
    #     #     self.templates["retro_template"][reaction_id])
    #     reaction = self.reactions[reaction_id]
    #     molecule = MolFromSmiles(state.molecules[idx])
    #     reactants = reaction.RunReactants([molecule])
    #     if len(reactants) > 0:
    #         reactants = reactants[0]
    #     else:
    #         return None # warning bad solution
    #     state.reactions[idx] = reaction_id
    #     for reactant in reactants:
    #         smiles = MolToSmiles(reactant)
    #         aizynth = Molecule(rd_mol = reactant)
    #         state.molecules.append(smiles)
    #         state.reactions.append(-1)
    #         try:
    #             in_stock = aizynth in STOCK
    #             state.in_stock.append(in_stock)
    #         except:
    #             return None
    #         state.children[idx].append(len(state.molecules)-1)
    #         state.children.append([])
    #         # Check for bad molecules
    #         if MolFromSmiles(smiles) == None:
    #             return None
    #     return state

    @lru_cache(maxsize=16_384)
    def reaction_mask(self, molecule_smiles: str) -> NDArray[np.bool]:
        mask: NDArray[np.bool] = np.full(len(self.reactions), True)
        molecule = MolFromSmiles(molecule_smiles)
        for i, reaction in enumerate(self.reactions):
            mask[i] = len(reaction.RunReactants([molecule])) == 0
        return mask

    def get_action_space(self):
        """
        Constructs a list with all possible actions, including eos.

        There are four types of actions. The main ones are one for choosing 
        which leaf to expand and another for choosing how to expand that leaf.
        The action types for selecting a target and stop are only used once.

        Actions are represented by tuples
        - The end-of-sequence (eos) action looks like this: (0, 0)
        - Target-selection actions look like this: `(1, tgt_idx)`
        - Leaf-selection actions look like this: `(2, idx)`
        - Expansion actions look like this: `(3, rxn_idx)`
        """
        actions: List[Tuple[int, int, int]] = []
        for tgt_idx in range(len(self.targets)):
            actions.append((ActionType.SELECT_TARGET, tgt_idx))
        for idx in range(self.max_n_nodes):
            actions.append((ActionType.SELECT_LEAF, idx))
        for rxn_idx in range(len(self.reactions)):
            actions.append((ActionType.EXPAND, rxn_idx))
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
        if state.n_reactions >= self.max_reactions:
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
        expansion_end = expansion_start + len(self.reactions)

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
        
        # If a leaf x is selected, we check which reactions can be applied to it.
        molecule: str = state[selected_leaf]["molecule"]
        reaction_mask = self.reaction_mask(molecule)
        relevant_slice = slice(expansion_start, expansion_end)
        mask[relevant_slice] = reaction_mask

        # If all other actions are invalid, make eos valid
        if all(mask):
            mask[-1] = False
        
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
        
        do_step, self.state, action = self._pre_step(action, skip_mask_check)
        if not do_step:
            return self.state, action, False
        # If action is eos
        if action == self.eos:
            self.done = True
            self.n_actions += 1
            return self.state, self.eos, True
        # If action is normal action, perform action
        action_type, action_value = action
        # Action type 1: target selection
        if action_type == ActionType.SELECT_TARGET:
            assert len(self.state.graph.nodes) == 0, "Target already selected."
            tgt_idx = action_value
            self.state.select_target(tgt_idx)
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
                warnings.warn("Leaf selection failed.")
                return self.state, action, False
        # Action type 3: expansion
        if action_type == ActionType.EXPAND:
            rxn_idx = action_value
            try:
                self.state.expand(rxn_idx)
                self.n_actions += 1
                return self.state, action, True
            except AssertionError:
                warnings.warn("Expansion failed.")
                return self.state, action, False
        # If the excution gets here something went wrong
        raise Exception("There is a bug if the control reaches here.")
        return self.state, action, False

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

        # If leaf is selected, the previous action was selecting that leaf.
        selected_leaf = state.get_selected_leaf()
        if selected_leaf != None:
            parent = state.copy()
            parent.unselect_leaf()
            return [parent,], [(ActionType.SELECT_LEAF, selected_leaf)]

        # If there is precisely one node in the tree, the previous action was 
        # selecting the target.
        if len(state.graph.nodes) == 1:
            target_value = state.graph.nodes[0]["molecule"]
            tgt_idx = state.targets.index(target_value)
            parent = state.copy()
            parent.unselect_target()
            return [parent,], [(ActionType.SELECT_TARGET, tgt_idx)]

        # If there was no selected leaf, the previous action was an expansion.
        parents: List[ReactionTree] = []
        actions: List[Tuple[int, int, int]] = []
        unexpandable = state.get_unexpandable()
        for idx in unexpandable:
            actions.append((ActionType.EXPAND, state[idx]["reaction"]))
            parent = state.copy()
            parent.unexpand(idx)
            parents.append(parent)
        return parents, actions

    def mol2tensor(self, molecule: str) -> TensorType["self.fingerprint_length"]:
        aizynthfinder_mol = Molecule(smiles = molecule)
        tensor = torch.tensor(
            aizynthfinder_mol.fingerprint(
                radius=2, 
                nbits=self.fingerprint_length
            ), 
            device = self.device
        )
        return tensor

    def mols2tensor(
        self, 
        molecules: List[str]
    ) -> TensorType["len(molecules)", "self.fingerprint_length"]:
        """
        Converts a molecule into a vector fingerprint.
        """
        fingerprint_length = 2048
        if len(molecules) == 0:
            return torch.empty((0, fingerprint_length))
        fingerprints = [self.mol2tensor(molecule) for molecule in molecules]
        return torch.stack(fingerprints)

    def state2tensor(
        self, 
        state: ReactionTree
    ) -> TensorType["max_n_nodes+1", "self.fingerprint_length+len(self.reactions)+4"]:
        mols = [value for _, value in state.graph.nodes(data="molecule")]
        mols_tensor = self.mols2tensor(mols)
        rxns = [value for _, value in state.graph.nodes(data="reaction")]
        # We transform reaction indexes to one hot vectors
        # The +1 is so that -1 goes to 0
        if rxns:
            rxns_one_hot = one_hot(torch.tensor(rxns) + 1, len(self.reactions) + 1)
        else:
            rxns_one_hot = torch.empty((0, len(self.reactions) + 1))
        in_stock = [value for _, value in state.graph.nodes(data="in_stock")]
        in_stock_tensor = torch.tensor(in_stock, dtype=torch.float32)
        leaf_nodes = torch.tensor(state.get_leaf_nodes())
        if leaf_nodes.shape[0] != 0:
            is_leaf = one_hot(leaf_nodes, num_classes=self.max_n_nodes).sum(axis=0)
        else:
            is_leaf = torch.zeros(self.max_n_nodes)
        active_leaf = state.get_selected_leaf()
        if active_leaf == None:
            is_active = torch.zeros(self.max_n_nodes)
            active_leaf_tensor = torch.zeros(
                self.fingerprint_length + len(self.reactions) + 4
            )
        else:
            is_active = one_hot(torch.tensor(active_leaf), self.max_n_nodes)
            active_leaf_one_hot = self.mols2tensor([state[active_leaf]["molecule"]])
            active_leaf_tensor = torch.cat(
                (
                    active_leaf_one_hot.squeeze(), 
                    torch.zeros(len(self.reactions) + 1),
                    torch.tensor([0, 1, 1])
                ))
        padding = (0, self.max_n_nodes - in_stock_tensor.shape[0])
        mols_padded = pad(mols_tensor, (0, 0) + padding, value=0)
        rxns_padded = pad(rxns_one_hot, (0, 0) + padding, value=0)
        in_stock_padded = pad(in_stock_tensor, padding, value=0).unsqueeze(1)
        is_leaf_unsqueezed = is_leaf.unsqueeze(1)
        is_active_unsqueezed = is_active.unsqueeze(1)
        result_tensor = torch.cat(
            (
                mols_padded,
                rxns_padded, 
                in_stock_padded, 
                is_leaf_unsqueezed, 
                is_active_unsqueezed,
            ), axis=-1).to(device=self.device)
        result_tensor_with_active_appended = torch.cat(
            (result_tensor, active_leaf_tensor.unsqueeze(0)), axis=0
        )
        return result_tensor_with_active_appended.to(dtype=torch.float32)

    def states2proxy(
        self, states: List[ReactionTree]
    ) -> List[ReactionTree]:
        return states

    def states2oracle(
        self, states: List[ReactionTree]
    ) -> List[ReactionTree]:
        return states

    def states2policy(
        self, states: List[ReactionTree]
    ) -> TensorType["batch_size", "policy_input_dim"]:
        return torch.stack(list(map(self.state2tensor, states)), axis = 0).flatten(1, -1)

    def state2readable(self, state: Optional[ReactionTree] = None):
        """
        Converts a state into a readable summary.
        """
        if state is None:
            state = self.state.copy()
        reactions = list(filter(lambda x: x != -1, state.graph.nodes(data="reaction")))
        leaf_mols_in_stock = []
        leaf_mols_not_in_stock = []
        for leaf_idx in state.get_leaf_nodes():
            if state[leaf_idx]["in_stock"]:
                leaf_mols_in_stock.append(state[leaf_idx]["molecule"])
            else:
                leaf_mols_not_in_stock.append(state[leaf_idx]["molecule"])
        leaf_mols_in_stock_str = ", ".join(leaf_mols_in_stock)
        leaf_mols_not_in_stock_str = ", ".join(leaf_mols_not_in_stock)
        reactions_str = ", ".join(map(str, reactions))
        return "\n".join([ 
            f"Reaction tree summary", 
            F"---------------------",
            f"Target: {state[0]['molecule']}", 
            f"Number of reactions: {len(reactions)}", 
            f"Reaction indices: {reactions_str}", 
            f"Molecules in stock ({len(leaf_mols_in_stock)}): "
            f"{leaf_mols_in_stock_str}", 
            f"Missing from stock ({len(leaf_mols_not_in_stock)}): "
            f"{leaf_mols_not_in_stock_str}",
        ])

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