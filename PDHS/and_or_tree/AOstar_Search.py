from Heuristics.Bounds import UpperBound, LowerBound

__author__ = 'Victor Szczepanski'

from PDHS.and_or_tree.AO_Tree import Tree, OR_Node

class AOStarSearcher(object):
    """The AO* search strategy.

    An AO* search strategy uses a heuristic to pick the next node in an `AO_Tree`'s fringe for expansion.

    The main interface is through the `forward_search` method. This will build an `AO_Tree` and search it until
    some condition is met - subclasses may define their own stopping conditions.



    """

    def __init__(self, expand_heuristic):
        self.expand_heuristic = expand_heuristic

    def forward_search(self, initial_belief_state: list, S: list, O: dict, T: dict, V: LowerBound,
                       upper_bound_function: UpperBound, stopping_function):
        """
        Using an initial belief state `initial_belief_state`, build and search an `AO_Tree` by repeatedly selecting a node
        from the `AO_Tree`'s fringe based on `expand_heuristic`. Stops after meeting some stopping condition.

        Note:
            We expect that the provided stopping condition is guaranteed to halt this search in finite steps.
            If the stopping condition fails to guarantee this, then `forward_search` may execute forever.

        Args:
            initial_belief_state (list[float]): a vector of probabilities that represent the initial belief state of the agent.
            S (list[str]): the states of the POMDP
            O (dict[str, dict[str, dict[str, float]]]): The probability function that the agent makes an observation w given an action a and next state s'.
            T (dict[str, dict[str, dict[str, float]]]): The probability function that the system changes from state s to state s' given an action a.
            stopping_function (function): A function reference that takes a single argument of type `Tree` and returns a Bool that represents whether to stop the forward search or not.
        Returns:
            AO_Tree: The state of the AO_Tree when the stopping condition was met.
        """
        tree = Tree(OR_Node(belief_state=initial_belief_state), upper_bound_function=upper_bound_function, lower_bound_function=V)
        while not stopping_function(tree):
            # select fringe node from tree using heuristic
            expand_node = self.expand_heuristic.select(tree.fringe) #TODO: In Ross, Pineau, et al., they calculate next heuristic in the expand function. Maybe it is more efficient that way?
            tree.expand(expand_node, O, T, S)
            print("Current size of tree: {}".format(tree.size()))
            # propagate upper and lower bounds

        return tree
