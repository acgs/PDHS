from Heuristics import Heuristics
from Models.models import GTModel, POMDPModel, PseudoPOMDPModel, Player
from and_or_tree import AOstar_Search
from and_or_tree.AO_Tree import Tree, OR_Node, Node

__author__ = 'Victor Szczepanski'


def main(input_filename: str):

    gtmodel = GTModel(input_filename)
    pomdp = POMDPModel(pseudo_pomdp_model=PseudoPOMDPModel(gtmodel=gtmodel))

    policy_driven_heuristic_search(pomdp=pomdp,beta=pomdp.discount, T=pomdp.state_transition, S=pomdp.states, O=pomdp.observation_probability)


def policy_driven_heuristic_search(pomdp, T, S, O, beta: float=0.0, epsilon: float=0.0):
    """Implement Hansen's heuristic search algorithm.

    We ignore step 1., since we get the finite-state controller and epsilon as arguments.
    2. Compute V for P.
    3. a) Forward search from starting belief state
    3. b) If error bound (difference of upper and lower bound) <= epsilon, exit
    3. c) Consider all reachable nodes in search tree that have lower bound improved, from leaves to root:
        3. c) i. If action and successor links are same as a machine state in `P`, keep it in P'.
        3. c) ii. Else if vector for this node pointwise dominates some state in `P`, change state in `P` to have the same action and successor links as this node.
        3. c) iii. Else add state to P' that has action and successor links as this node.
    3. d) Prune any state in P' that is not reachable from state that optimizes starting belief
    4. Set P = P'. If 3. c) ii. changed a state, goto 2. Else goto 3.

    :param P:
    :param T:
    :param S:
    :param O:
    :param beta:
    :param epsilon:
    :return:
    """
    P = pomdp.players[0]
    searcher = AOstar_Search.AOStarSearcher(Heuristics.HansenHeuristic(beta=beta))

    skip_compute_V = False
    while True:
        # step 2.
        if not skip_compute_V:
            V = pomdp.to_value_function(P)

        # step 3. a)
        tree = searcher.forward_search(initial_belief_state=[], S=S, T=T, O=O)
        """:type : Tree"""
        # step 3. b)
        if tree.root.upper_bound - tree.root.lower_bound <= epsilon:
            break

        # step 3. c)
        tree_stack = [tree.root]
        tree_visited = []
        while len(tree_stack) is not 0:
            node = tree_stack.pop()
            if node.lower_bound is not node.initial_lower_bound:
                tree_visited.append(node)
            for child in node.children:
                tree_stack.append(child)

        # order the nodes by depth
        sorted_nodes = sorted(tree_visited, key=_sort_by_depth)

        for node in sorted_nodes:
            # step 3. c) i

            # step 3. c) ii

            #here we'll set skip_compute_V, instead of waiting for step 4.

            # step 3. c) iii
            pass

        # step 3. d)

        # step 4.


def _sort_by_depth(node: Node):
    """
    Function for sorting nodes by depth
    Args:
        node (Node): the node to get its key to sort by - i.e. the node to get its depth.
    Returns:
        node.depth
    """
    return node.depth