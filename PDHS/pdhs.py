import argparse
import time
from Heuristics import Heuristics, Bounds
from Heuristics.Bounds import LBValueFunction
from Models.models import GTModel, POMDPModel, PseudoPOMDPModel, Player
from and_or_tree import AOstar_Search
from and_or_tree.AO_Tree import Tree, Node

__author__ = 'Victor Szczepanski'


def main(input_filename: str, verbose: bool):

    gtmodel = GTModel(input_filename)
    pomdp = POMDPModel(pseudo_pomdp_model=PseudoPOMDPModel(gt_model=gtmodel))

    print("POMDP:\n{}".format(pomdp))

    # We'd like T to be a dictionary, not a list of tuples
    T = {}
    for s in pomdp.states:
        T[s] = {}
        for a in pomdp.actions:
            T[s][a] = {}
            for s_prime in pomdp.states:
                T[s][a][s_prime] = 0

    for (s_prime, (s, a), t) in pomdp.state_transition:
        T[s][a][s_prime] = float(t)

    upper_bound = Bounds.UB_MDP(S=pomdp.states, A=pomdp.actions, T=T, R=pomdp.payoff, gamma=float(pomdp.discount))

    #train the upper bound before using it.
    train_upper_bound(upper_bound=upper_bound)

    policy_driven_heuristic_search(pomdp=pomdp, beta=pomdp.discount,
                                   T=T, S=pomdp.states, O=pomdp.observation_probability,
                                   upper_bound=upper_bound, initial_belief_state=[1, 0, 0, 0])


def train_upper_bound(upper_bound: Bounds.ImprovableUpperBound, timeout: float=10, max_iterations: int=100):
    """
    Simply runs the upper bound's improve function iteratively until it converges or runs out of time.

    Args:
        upper_bound (UpperBound): the upper bound to train/improve
        timeout (float): the time, in seconds, to allow training to occur.
        max_iterations (int): the maximum number of iterations to let the training run.
    """
    start_time = time.time()
    while time.time() < start_time + timeout and upper_bound.improve_iterations < max_iterations:
        old_value = upper_bound.value([1, 0, 0, 0])
        upper_bound.improve()
        new_value = upper_bound.value([1, 0, 0, 0])
        print("Difference from old to new: {}".format(new_value-old_value))
        print("New: {}".format(new_value))
        if upper_bound.is_converged:
            break


def policy_driven_heuristic_search(pomdp, T, S, O, upper_bound: Bounds.UpperBound, initial_belief_state: list=[],
                                   beta: float=0.0, epsilon: float=0.0):
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
    exit_condition = tree_exit_condition(epsilon=epsilon)

    skip_compute_V = False
    while True:
        # step 2.
        if not skip_compute_V:
            V = LBValueFunction()
            V.from_Player(player=pomdp.players[0], pomdp=pomdp)

        # step 3. a)
        tree = searcher.forward_search(initial_belief_state=initial_belief_state,
                                       S=S, T=T, O=O, V=V,
                                       upper_bound_function=upper_bound,
                                       R=pomdp.payoff, stopping_function=exit_condition)
        """:type : Tree"""
        # step 3. b)
        if tree.root.upper_bound - tree.root.lower_bound <= epsilon:
            break

        # step 3. c)
        tree_stack = [tree.root]
        tree_visited = []
        while len(tree_stack) is not 0:
            node = tree_stack.pop()
            print(node)
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


class tree_exit_condition(object):
    """Define an exit condition for forward search of an AO-Tree.

    In particular, implement the condition described by Hansen (1997) -
    if lower bound of root is improved or difference of upper and lower bound of root is <= epsilon, exit.

    Attributes:
        epsilon (float): the value for epsilon convergence.

    """
    def __init__(self, epsilon:float):
        self.epsilon = epsilon

    def stop(self, t: Tree):
        """Stop forward search if lower bound of root is improved or error bound is less than or equal to epsilon.

        Args:
            t (Tree): The AO Tree to test if forward search can stop on it.

        Returns:
            (bool): True if the stopping condition is met. False otherwise.
        """
        if t.root.initial_lower_bound != t.root.lower_bound or t.root.upper_bound - t.root.lower_bound <= self.epsilon:
            return True
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parses a Game Theory model and runs Policy Driven Heuristic Search')
    parser.add_argument('gtmodel', type=str, help='The input file name for the Game Theory model.')

    parser.add_argument('-verbose', type=bool, help='Verbosity of output.'
                                                    ' If true, will output in verbose mode.', default=False)

    #TODO: Add arguments for heuristic and upper bound selections.

    args = parser.parse_args()
    print(args.verbose)
    main(args.gtmodel, args.verbose)
