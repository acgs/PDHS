import argparse
import time
from Heuristics import Heuristics, Bounds
from Heuristics.Bounds import LBValueFunction
from Models.models import GTModel, POMDPModel, PseudoPOMDPModel, Player
from and_or_tree import AOstar_Search
from and_or_tree.AO_Tree import Tree, Node, OR_Node, AND_Node

__author__ = 'Victor Szczepanski'


def main(input_filename: str, verbose: bool):

    gtmodel = GTModel(input_filename)
    pomdp = POMDPModel(pseudo_pomdp_model=PseudoPOMDPModel(gt_model=gtmodel))

    if verbose:
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
    if verbose:
        print("POMDP's state transition: {}".format('\n'.join([ '({}, ({}, {}), {})'.format(s_next, s, a, t) for (s_next, (s, a), t) in pomdp.state_transition])))
        print("Created T -> mapping from state to action to next state to probability:")
        for s in pomdp.states:
            for a in pomdp.actions:
                for s_prime in pomdp.states:
                    print("T[{}][{}][{}]: {}".format(s, a, s_prime, T[s][a][s_prime]))

    upper_bound = Bounds.UB_MDP(S=pomdp.states, A=pomdp.actions, T=T, R=pomdp.payoff, gamma=float(pomdp.discount))

    #train the upper bound before using it.
    train_upper_bound(upper_bound=upper_bound)

    qmdp_upper_bound = Bounds.UB_QMDP(S=pomdp.states, A=pomdp.actions, T=T, R=pomdp.payoff, gamma=float(pomdp.discount),initial_V_MDP=upper_bound)

    train_upper_bound(upper_bound=qmdp_upper_bound)

    fib_upper_bound = Bounds.UB_FIB(R=pomdp.payoff, gamma=float(pomdp.discount), A=pomdp.actions, O=pomdp.observation_probability, S=pomdp.states, T=T, Z=pomdp.observations, converged_QMDP=qmdp_upper_bound.Q_MDP)

    train_upper_bound(upper_bound=fib_upper_bound)

    policy_driven_heuristic_search(pomdp=pomdp, beta=pomdp.discount, observations=pomdp.observations, A=pomdp.actions,
                                   T=T, S=pomdp.states, O=pomdp.observation_probability,
                                   upper_bound=fib_upper_bound, initial_belief_state=[1, 0, 0, 0])


def train_upper_bound(upper_bound: Bounds.ImprovableUpperBound, timeout: float=10, max_iterations: int=100000):
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
        if upper_bound.is_converged:
            break


def policy_driven_heuristic_search(pomdp, observations, A, T, S, O, upper_bound: Bounds.UpperBound, initial_belief_state: list=[],
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
    P = pomdp.player1
    print("Other player's policy: {}".format(pomdp.players[-1]))
    """:type: Player"""

    print("Initial policy: {}".format(P))

    print("Actions: {}".format(A))
    print("Observations: {}".format(observations))
    print("O: {}".format(O))

    searcher = AOstar_Search.AOStarSearcher(Heuristics.HansenHeuristic(beta=beta))
    exit_condition = tree_exit_condition(epsilon=epsilon)

    new_state_counter = 0

    skip_compute_V = False
    while True:
        # step 2.
        if not skip_compute_V:
            V = LBValueFunction()
            V.from_Player(player=P, pomdp=pomdp)
            skip_compute_V = True

        # step 3. a)
        tree = searcher.forward_search(initial_belief_state=initial_belief_state, gamma=beta, A=A,
                                       S=S, observations=observations, T=T, O=O, V=V,
                                       upper_bound_function=upper_bound,
                                       R=pomdp.payoff, stopping_function=exit_condition)
        """:type : Tree"""
        # step 3. b)
        if tree.root.upper_bound - tree.root.lower_bound <= epsilon:
            break

        #for node in tree.fringe:
            #print(node)

        # step 3. c)

        # we search from the root, choosing the OR nodes that optimize the lower bound

        # we only want AND_Nodes, since they tell us the action and their children are the successor links
        # (belief of child maps to an alpha-vector/state in V/player1's machine)
        tree_stack = [tree.root]
        tree_visited = []
        #print("Searching for reachable nodes from root.")
        while len(tree_stack) is not 0:
            node = tree_stack.pop()
            #print("Expanding {} {}".format(str(type(node)), node))
            # For some reason isinstance doesn't want to work properly on node. Not sure why this is.
            # Workaround (very ugly!) is to look for specific strings in node's type.
            #print("{} {} {}".format(type(node), 'AND_Node' in str(type(node)), 'OR_Node' in str(type(node))))

            #we select the AND_Node child of node that optimizes the lower bound
            best_and_children = []
            best_lower_bound = 0
            for child in node.children:

                if child.lower_bound > best_lower_bound:
                    best_lower_bound = child.lower_bound
                    best_and_children = [child]
                elif child.lower_bound == best_lower_bound:
                    best_and_children.append(child)
                #if child.lower_bound != child.initial_lower_bound:
                    #best_and_children.append(child)
            print("Selected children: {}".format(best_and_children))
            tree_visited.extend(best_and_children)
            for and_child in best_and_children:
                tree_stack.extend(and_child.children)

            # if 'AND_Node' in str(type(node)) and node.lower_bound is not node.initial_lower_bound:
            #     #print("Visiting {}".format(node.belief_state))
            #     tree_visited.append(node)
            #     for child in node.children:
            #         tree_stack.append(child)
            # elif 'OR_Node' in str(type(node)):
            #     for child in node.children:
            #         #print("Adding {}".format(child))
            #         tree_stack.append(child)

        # order the nodes by depth
        sorted_nodes = sorted(tree_visited, key=_sort_by_depth)
        """:type: list[AND_Node]"""
        #print("Stopping loop. Tree's root: {}".format(tree.root))
        #print("Epsilon: {}".format(epsilon))
        #print("Reachable nodes whose values changed: {}".format(sorted_nodes))

        # we create a new player to represent the new policy/FSA
        P_prime = Player(name=P.name, actions=P.actions, signals=P.signals,
                         observation_marginal_distribution=P.observation_marginal_distribution, payoff=P.payoff)
        V_prime = LBValueFunction(alpha_vectors=V.alpha_vectors, actions=V.actions,
                                  states=V.pomdp_states, machine_states=V.machine_states)
        #V_prime = LBValueFunction()

        for node in sorted_nodes:
            """:type: AND_Node"""
            print("Iterating over node {}".format(node))

            #for each node, calculate the corresponding vector based on the current value function - Hansen 1998, pg. 217


            # calculate successor state from belief of each child of node
            successor_states = {}
            for child in node.children:
                try:
                    alpha_vector = V.best_alpha_vector(child.belief_state) # I think we have to use V_prime here, since we might update it every iteration
                    successor_state = V_prime.machine_states[V.alpha_vectors.index(alpha_vector)]
                    successor_states[child.observation] = successor_state
                except ValueError:
                    continue
            print("Found successors: {}".format(successor_states))
            # step 3. c) i
            #for some state s in player1.state_transitions and player1.state_machine has the same action and transitions as node, keep those transitions in new_player
            kept_state = False
            for s in P.states:
                if node.action == P.state_machine[s]:
                    different_successor_found = False

                    #are successors of node the same as s?
                    for observation, successor in successor_states.items():
                        if P.state_transitions[s][observation] != successor:
                            different_successor_found = True
                            break

                    if different_successor_found:
                        print("Found that node has different successor states than {}: g->{}, b->{}".format(s, P.state_transitions[s]['g'], P.state_transitions[s]['b']))
                        continue

                    #keep s in P_prime
                    kept_state = True
                    print("Found that node has the same successor states as some state in last policy. Keeping it in new policy.")
                    P_prime.add_machine_state(s, node.action, successor_states)

            changed_state = False
            if not kept_state:
                # step 3. c) ii

                # compute alpha-vector for this node by finding the value of this node at each pomdp state
                # \alpha_{t+1}^a(s) = R(s,a) + gamma sum_{s' in S} T(s,a,s')V_t-1(s')

                # Milos defines next alpha vector as:
                # \alpha_{b,a}^{i+1} = \rho(s,a) + \sum_{o \in O} \sum_{s' \in S} P(s', o | s, a) * \alpha_i^{\iota(b,a,o)}(s')
                # where \iota(b,a,o) indexes a linear vector \alpha_i in old alpha vector set that maximizes:
                # \sum_{s' \in S} ( \sum_{s \in S} P(s', o | s, a)*b(s) ) \alpha_i(s')

                # I think they are the same definition: R = \rho, T = P, and V_{t-1} = \alpha_i^{\iota(b,a,o)}(s').
                # But not sure who Milos requires summing over observations.

                alpha_vector = []
                for s in pomdp.states:
                    sum_s = 0
                    for s_prime in pomdp.states:
                        sum_s += T[s][node.action][s_prime] * V.value_at_state(state=s_prime)

                    alpha_vector.append(float(pomdp.payoff[(node.action, s)]) + float(beta) * sum_s)
                pass

                # test for pointwise dominance of any alpha-vector in V
                for old_alpha_vector in V.alpha_vectors:
                    # if it dominates, change action/successor links
                    if pointwise_dominantes(alpha_vector1=alpha_vector, alpha_vector2=old_alpha_vector):
                        skip_compute_V = False

                        s = V.machine_states[V.alpha_vectors.index(old_alpha_vector)]
                        print("Found that new vector {} dominates old vector {}. Changing state in new Policy.".format(alpha_vector, old_alpha_vector))
                        print("Before change: {}".format(P_prime))
                        P_prime.remove_machine_state(s)
                        P_prime.add_machine_state(s, node.action, successor_states)
                        print("Current P': {}".format(P_prime))
                        changed_state = True
                        break

            # step 3. c) iii
            if not changed_state:
                skip_compute_V = False
                new_state = ''.join(['new_state', str(new_state_counter)])
                print("Found a new state to improve policy. Adding state {} with successors {} and action {}".format(new_state, successor_states, node.action))
                new_state_counter += 1
                # create new machine state
                P_prime.add_machine_state(new_state, node.action, successor_states)
                print("Current new policy: {}".format(P_prime))

        # step 3. d)
        # search for states that are unreachable from the machine state that optimizes starting belief in P_prime
        V_prime = V_prime.from_Player(player=P_prime, pomdp=pomdp)
        print("Looking for best vector for belief state {}.".format(initial_belief_state))
        print("Available vectors: {}".format(V_prime.alpha_vectors))
        best_vector = V_prime.best_alpha_vector(belief_state=initial_belief_state)
        best_state = V_prime.machine_states[V_prime.alpha_vectors.index(best_vector)]

        #collect all machine states that are reachable using DFS
        reachable_stack = [best_state]
        print("Starting pruning from state {}".format(best_state))
        reached = []
        reached_nodes = []
        while len(reachable_stack) is not 0:
            state = reachable_stack.pop()
            print("Keeping state {}. It has transitions {} and action {}".format(state, P_prime.state_transitions[state], P_prime.state_machine[state]))
            reached.append((state, P_prime.state_transitions[state], P_prime.state_machine[state]))
            reached_nodes.append(state)
            for o in observations:
                next_state = P_prime.state_transitions[state][o]
                if next_state not in reached_nodes:
                    reachable_stack.append(next_state)

        #rebuild P_prime from reached nodes
        P_prime.states = []
        P_prime.state_transitions = {}
        P_prime.state_machine = {}

        for (reached_node, state_transitions, action) in reached:
            P_prime.add_machine_state(reached_node, action, state_transitions)

        # step 4.
        P = P_prime
        print("New P: {}".format(P))


def pointwise_dominantes(alpha_vector1: list, alpha_vector2: list):
    """Test if `alpha_vector1` pointwise dominates `alpha_vector2`.

    Note:
        We assume that `alpha_vector1` does not dominate `alpha_vector2` if they are identical.
    Args:
        alpha_vector1 (list[float]): an alpha-vector of a ValueFunction to test if dominates.
        alpha_vector2 (list[float]): an alpha-vector of a ValueFunction to test if dominated.
    Returns:
        True if `alpha_vector1` pointwise dominates `alpha_vector2`.
    """
    assert(len(alpha_vector1) == len(alpha_vector2))
    if alpha_vector1 == alpha_vector2:
        return False
    for a1, a2 in zip(alpha_vector1, alpha_vector2):
        if a2 > a1:
            return False
    return True


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
        print("root's upper bound: {}, lower bound: {}, initial upper bound: {}, initial lower bound: {}."
              .format(t.root.upper_bound, t.root.lower_bound, t.root.initial_upper_bound, t.root.initial_lower_bound))
        largest_depth = 0
        lowest_depth = 1000000000
        for n in t.fringe:
            if n.depth > largest_depth:
                largest_depth = n.depth
            if n.depth < lowest_depth:
                lowest_depth = n.depth
        print("Deepest node's depth: {}, shallowest node's depth: {}".format(largest_depth, lowest_depth))
        if t.root.initial_lower_bound != t.root.lower_bound:
            print("Improved root's lower bound. Root initial bound: {}. Current bound: {}. Epsilon: {}".format(t.root.initial_lower_bound, t.root.lower_bound, self.epsilon))
            return True
        elif t.root.upper_bound - t.root.lower_bound <= self.epsilon:
            print("Error bound converged to epsilon.")
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
