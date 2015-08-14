from Heuristics.Bounds import LowerBound, UpperBound, Bound
from and_or_tree.utils import memoize

__author__ = 'Victor Szczepanski'

import sys

class Tree(object):
    """Represent an AND/OR Tree.

    Args:
        root (OR_Node): the root node of this AND/OR Tree.
        actions (list[str]): the list of actions we can expand from an OR_Node.
        observations (list[str]): the list of observations we can expand from an AND_Node.

    Attributes:
        root (OR_Node): the root node of this AND/OR Tree.
        U (UpperBound): the upper bound function to use for fringe nodes
        L (LowerBound): the lower bound function to use for fringe nodes
        actions (list[str]): the list of actions we can expand from an OR_Node.
        observations (list[str]): the list of observations we can expand from an AND_Node.
        fringe (list[OR_Node]): The fringe nodes that can be chosen for expansion.
    """

    def __init__(self, root, upper_bound_function: UpperBound, lower_bound_function: LowerBound, actions: list, observations: list):
        self.root = root
        self.actions = actions
        self.observations = observations
        self.fringe = [self.root]
        self.fringe_beliefs = [self.root.belief_state]
        self.U = upper_bound_function
        self.L = lower_bound_function

    def size(self):
        """Calculate the size of this Tree.

        Returns the sum of the sizes of the nodes in this tree, plus the actions and observations.

        :return:
        """
        size = 0
        # iterate over nodes from root, using a stack
        node_stack = [self.root]
        while len(node_stack) is not 0:
            n = node_stack.pop()
            """:type : Node"""
            size += n.size()
            for child in n.children:
                node_stack.append(child)

        size += sys.getsizeof(self.actions) + sys.getsizeof(self.observations)
        return size

    def expand(self, node, gamma, O: dict, T: dict, S: list, R: dict):
        r"""Expand a given node in this tree.

        Expands a node as described in "Online Planning Algorithms for POMDPs" by Ross, Pineau, et. al, Journal of Artifical Intelligence Research 32, (2008), 663-704.

        In particular, implements the function

        .. math::
            b_t(s') = \tau(b_{t-1}, a_{t-1}, z_t)(s') = \frac{1}{Pr(z_t | b_{t-1}, a_{t-1})} * O(s', a_{t-1}, z_t)
            \sum_{s \in S} T(s, a_{t-1}, s')b_{t-1}(s)

            Pr(z | b, a) = \sum_{s' \in S} O(s', a, z) \sum_{s \in S} T(s, a, s')*b(s)

        Note that `node` is always an OR-node, because after we expand it to all actions (AND_Node),
        and then immediately expand the actions with every possible observation.

        ..precondition::
            len(node.children) is 0

        ..postcondition::
            len(node.children) is not 0
            for node in self.fringe:
                len(node.children) is 0

        Args:
            node (OR_Node): the node in this tree to expand.
            O (dict[str, dict[str, dict[str, float]]]): The observation function that maps a state to an action to an observation to a probability. Represents the conditional probability of observing w given state s and action a.
            T (dict[str, dict[str, dict[str, float]]]): The transition function that maps a state to an action to a state to a probability. Represents the conditional probability of transitioning from state s to s' given action a.
            S (list[str]): The states of the POMDP.
            R (Dict[tuple[ACTION,STATE], float]): Mapping from action/state tuple to a payoff real value. Represents the immediate payoff of taking an action in a state.
        """
        gamma = float(gamma)
        # First remove node from fringe.
        try:
            self.fringe.remove(node)
        except ValueError:
            raise ValueError("Cannot expand a node that is not in the fringe of the tree.")

        try:
            self.fringe_beliefs.remove(node.belief_state)
        except ValueError:
            raise ValueError("Cannot expand a node whose belief state is not in the fringe.")

        #Generate AND_Nodes for every action
        and_children = []
        """:type : list[AND_Node]"""

        print("Expanding OR_Node with belief state {}".format(node.belief_state))

        for action in self.actions:
            print("Adding AND_Node with action {}".format(action))
            new_child = AND_Node(parent=node, action=action, belief_state=node.belief_state)
            and_children.append(new_child)
        # for each new AND_Node, create an OR_Node and compute its belief state according to:
        # b_t(s') = \tau(b_{t-1}, a_{t-1}, z_t)(s')
        for and_child in and_children: #iterate over all new AND_Nodes.
            print("Expanding AND_Node with action {}".format(and_child.action))
            a = and_child.action
            for observation in self.observations:  # this generates each child of the and_child
                new_belief_state = self.tau(belief_state=node.belief_state, action=a, observation=observation, O=O, T=T, S=S)
                obs_prob = self.pr_z(observation=observation, belief_state=node.belief_state, action=a,
                                     O=O, T=T, S=S)

                # probability of reaching new_child is the probability of reaching node * Pr(z | b,a)
                reach_probability = node.reach_probability*obs_prob

                new_child = OR_Node(parent=and_child, belief_state=new_belief_state, obs_prob=obs_prob,
                                    reach_probability=reach_probability, lower_bound=self.L.value(new_belief_state),
                                    upper_bound=self.U.value(new_belief_state), observation=observation)
                print("New OR_Node child with belief state: {}".format(new_child.belief_state))
                self.fringe.append(new_child)
                self.fringe_beliefs.append(new_child.belief_state)
                #print("New fringe: {}".format(self.fringe))
                #print("New fringe beliefs: {}".format(self.fringe_beliefs))

        #We propagate after all children are created, so that we don't propagate information from fringe nodes that we only propagate once for each fringe node.
        fringe_upper_bounds = [n.upper_bound for n in self.fringe]
        fringe_lower_bounds = [n.lower_bound for n in self.fringe]
        self.propagate(gamma, R, O, T, S, fringe_nodes=self.fringe)
        assert fringe_lower_bounds == [n.lower_bound for n in self.fringe]
        assert fringe_upper_bounds == [n.upper_bound for n in self.fringe]

    def propagate(self, gamma, R, O, T, S, fringe_nodes: list=[]):
        r"""Propagate upper and lower bound information to root.

        Updates upper and lower bounds for ancestors of nodes in `fringe_nodes`.

        Args:
            R (Dict[tuple[ACTION,STATE], float]): Mapping from action/state tuple to a payoff real value. Represents the immediate payoff of taking an action in a state.
            O (dict[str, dict[str, dict[str, float]]]): The observation function that maps a state to an action to an observation to a probability. Represents the conditional probability of observing w given state s and action a.
            T (dict[str, dict[str, dict[str, float]]]): The transition function that maps a state to an action to a state to a probability. Represents the conditional probability of transitioning from state s to s' given action a.
            S (list[str]): The states of the POMDP.
            fringe_nodes Optional(list[OR_Node]): the nodes on the fringe from which to start bound propagation.
        """
        for fringe_node in fringe_nodes:
            node_ptr = fringe_node.parent
            """:type: Node"""
            while node_ptr is not None:
                self.update_lower_bound(node_ptr, gamma, R=R, O=O, T=T, S=S)
                self.update_upper_bound(node_ptr, gamma, R=R, O=O, T=T, S=S)
                node_ptr = node_ptr.parent

    def update_lower_bound(self, node, gamma, R, O, T, S):
        r"""Update the lower bound of `node`.

        We propagate the lower bounds of the fringe nodes up the tree with these equations:

        .. math::
            L_T(b) = \begin{cases} L(b), & \text{if } b \in F(T) \\ \text{max}_{a \in A} L_T(b,a), & \text{otherwise}\end{cases}

            L_T(b,a) = R_B(b,a) + \gamma \sum_{z \in Z} \text{Pr}(z | b,a)L_T(\tau(b,a,z))


        F(T) is the set of fringe nodes in tree T, L_T(b) represents the lower bound on V'(b) associated to belief state b,
        L_T(b,a) represents the corresponding bound on Q*(b,a), and L(b) is the bound of a fringe node.


        Args:
            node (Node): the node whose lower bound should be updated
            R (Dict[tuple[ACTION,STATE], float]): Mapping from action/state tuple to a payoff real value. Represents the immediate payoff of taking an action in a state.
            O (dict[str, dict[str, dict[str, float]]]): The observation function that maps a state to an action to an observation to a probability. Represents the conditional probability of observing w given state s and action a.
            T (dict[str, dict[str, dict[str, float]]]): The transition function that maps a state to an action to a state to a probability. Represents the conditional probability of transitioning from state s to s' given action a.
            S (list[str]): The states of the POMDP.
        """
        if 'OR_Node' in str(type(node)) and node not in self.fringe:
            print("Updating internal OR_Node lower bound as max of {}".format([child.lower_bound for child in node.children]))
            node.lower_bound = max([child.lower_bound for child in node.children])
        else:
            #node.lower_bound = self.bound_t(node.belief_state, gamma, R, O, T, S, self.L)
            node.lower_bound = self.bound_t_new(node, gamma, R, O, T, S, self.L)

    def update_upper_bound(self, node, gamma, R, O, T, S):
        r"""Update the upper bound of `node`.

        We propagate the upper bounds of the fringe nodes up the tree with these equations:

        .. math::
            U_T(b) = \begin{cases}U(b), & \text{if } b \in F(T) \\ \text{max}_{a \in A} U_T(b,a), & \text{otherwise}\end{cases}

            U_T(b,a) = R_B(b,a) + \gamma \sum_{z \in Z} \text{Pr}(z | b,a)U_T(\tau(b,a,z))

        F(T) is the set of fringe nodes in tree T, U_T(b) represents the upper bound on V'(b) associated to belief state b,
        U_T(b,a) represents the corresponding bound on Q*(b,a), and U(b) is the bound of a fringe node.

        Args:
            node (Node): the node whose upper bound will be updated.
            R (Dict[tuple[ACTION,STATE], float]): Mapping from action/state tuple to a payoff real value. Represents the immediate payoff of taking an action in a state.
            O (dict[str, dict[str, dict[str, float]]]): The observation function that maps a state to an action to an observation to a probability. Represents the conditional probability of observing w given state s and action a.
            T (dict[str, dict[str, dict[str, float]]]): The transition function that maps a state to an action to a state to a probability. Represents the conditional probability of transitioning from state s to s' given action a.
            S (list[str]): The states of the POMDP.

        """
        if 'OR_Node' in str(type(node)) and node not in self.fringe:
            print("Updating internal OR_Node upper bound as max of {}".format([child.upper_bound for child in node.children]))
            node.upper_bound = max([child.upper_bound for child in node.children])
        else:
            #node.upper_bound = self.bound_t(node.belief_state, gamma, R, O, T, S, self.U)
            node.upper_bound = self.bound_t_new(node, gamma, R, O, T, S, self.U)


    def bound_t_new(self, node, gamma, R, O, T, S, fringe_function: Bound):
        """
        Implement bound calculation relative to nodes.

        In the usual calculation, we define:

        Note:
            :math:`B_T(b)` is implemented implicitly in the ``update_x_bound`` functions, so we just implement B_T(b,a) here.

        .. math::
            B_T(b) = \begin{cases}B(b), & \text{if } b \in F(T) \\ \text{max}_{a \in A} B_T(b,a), & \text{otherwise}\end{cases}

            B_T(b,a) = R_B(b,a) + \gamma \sum_{z \in Z} \text{Pr}(z | b,a)B_T(\tau(b,a,z))

        However, note that the belief of a node's child is exactly :math:`\tau(b,a,z)`
        Thus, :math:`B_T(\tau(b,a,z))` is just a lookup of the bound for the zth child of this (AND) Node.

        So, we'll implement B_T(b,a) as:

        .. math::
            B_T(node) = R_B(node.b,node.action) + \gamma \sum_{child \in node.children} \text{Pr}(child.observation | node.b,node.action)*child.bound

        :param node:
        :param gamma:
        :param R:
        :param O:
        :param T:
        :param S:
        :param fringe_function:
        :return:
        """
        if node in self.fringe:
            return fringe_function.value(node.belief_state)

        assert('AND_Node' in str(type(node)))
        discounted_sum = 0
        for child in node.children:
            child_bound = 0
            if isinstance(fringe_function, UpperBound):
                child_bound = child.upper_bound
            elif isinstance(fringe_function, LowerBound):
                child_bound = child.lower_bound
            else:
                raise ValueError("fringe_function to bound_t_new is neither {} nor {}, but {}.".format(UpperBound, LowerBound, type(fringe_function)))
            discounted_sum += self.pr_z(child.observation, node.belief_state, node.action, O, T, S) * child_bound
        discounted_sum *= gamma
        # We calculate R_B(belief_state, action) here, from R.
        R_B = self.R_B(belief_state=node.belief_state, action=node.action, S=S, R=R)
        return R_B + discounted_sum


    def bound_t(self, belief_state, gamma, R, O, T, S, fringe_function: Bound):
        """Abstract update to upper bound and lower bound.

        By taking a function parameter, abstract the calculation for bound propagation.

        If `belief_state` is a fringe belief, we use `fringe_function` to calculate its bound.
        Thus, if `fringe_function` is a ``Lower_Bound``, `bound_t` calculates the lower bound.

        Args:
            belief_state (list[float]): the belief state to use for updating
            R (Dict[tuple[ACTION,STATE], float]): Mapping from action/state tuple to a payoff real value. Represents the immediate payoff of taking an action in a state.
            O (dict[str, dict[str, dict[str, float]]]): The observation function that maps a state to an action to an observation to a probability. Represents the conditional probability of observing w given state s and action a.
            T (dict[str, dict[str, dict[str, float]]]): The transition function that maps a state to an action to a state to a probability. Represents the conditional probability of transitioning from state s to s' given action a.
            S (list[str]): The states of the POMDP.
            fringe_function (Bound): A Bound that exposes a value method to use for fringe node Bound calculation.

        Returns:
            float: The new bound for the node corresponding to the belief state `belief_state`.
        """
        if belief_state in self.fringe_beliefs:
            return fringe_function.value(belief_state)
        else:
            new_bound = 0
            for action in self.actions:
                bound_t = self.bound_action(belief_state=belief_state, gamma=gamma,
                                            R=R, O=O, T=T, S=S, action=action, fringe_function=fringe_function)
                if bound_t > new_bound:
                    new_bound = bound_t

        return new_bound

    def bound_action(self, belief_state, gamma, R, O, T, S, action, fringe_function):
        r"""
        Calculate the sub-function used by bound_t.

        Implement:

        .. math::
            B_T(b,a) = R_B(b,a) + \gamma \sum_{z \in Z} \text{Pr}(z | b,a)B_T(\tau(b,a,z))

            R_B(beliefstate, action) = \sum_{s \in S} b(s)R(s,a)


        Where B_T is a generic bound - the true bound is given by `fringe_function` -
        if it is a lower bound, calculates L_T, if it is an upper bound, calculates U_T.

        Args:
            belief_state (list[float]): the belief state to use for updating
            R (Dict[tuple[ACTION,STATE], float]): Mapping from action/state tuple to a payoff real value. Represents the immediate payoff of taking an action in a state.
            O (dict[str, dict[str, dict[str, float]]]): The observation function that maps a state to an action to an observation to a probability. Represents the conditional probability of observing w given state s and action a.
            T (dict[str, dict[str, dict[str, float]]]): The transition function that maps a state to an action to a state to a probability. Represents the conditional probability of transitioning from state s to s' given action a.
            S (list[str]): The states of the POMDP.
            action (str): The action to calculate the bound with.
            fringe_function (Bound): A Bound that exposes a value method to use for fringe node Bound calculation.
        Returns:
            float: B_T(`belief_state`, `action`) - the bound for this `belief_state`, `action` pair.
        """
        weighted_lower_bound = 0
        for observation in self.observations:
            weighted_lower_bound += (self.pr_z(observation=observation, belief_state=belief_state,
                                               action=action, O=O, S=S, T=T) *
                                     self.bound_t(self.tau(belief_state=belief_state, action=action,
                                                           observation=observation, O=O, T=T, S=S),
                                                  gamma, R, O, T, S, fringe_function=fringe_function)
                                     )

        # We calculate R_B(belief_state, action) here, from R.
        R_B = self.R_B(belief_state=belief_state, action=action, S=S, R=R)
        return R_B + float(gamma)*weighted_lower_bound


    @memoize
    def R_B(self, belief_state, action, S, R):
        return sum([belief*float(R[(action, S[i])]) for i, belief in enumerate(belief_state)])

    @memoize
    def pr_z(self, observation: str, belief_state: list, action: str, O: dict, T: dict, S: list):
        r"""Implement the Pr(z | b, a) function.

        Implements the function:

        .. math::
            Pr(z | b, a) = \sum_{s' \in S} O(s', a, z) \sum_{s \in S} T(s, a, s')*b(s)

        Args:
            observation (str): the observation z
            belief_state (list[float]): the belief state b
            action (str): the action a
            O (dict[str, dict[str, dict[str, float]]]): the observation probability function. Function of state, action, observation.
            T (dict[ str, dict[str, dict[str, float]]]): the state transition probability function. Function of state, action, state.
            S (list[str]): the state set.
        Returns:
            float: the probability of observing `observation` given belief state `belief_state` and action `action`.
        """
        z = observation
        b = belief_state
        a = action
        #calculate Pr(z | b,a)
        pr_z = 0.0
        for s_prime in S:
            new_observation = float(O[z][(a, s_prime)])#[s_prime][a][z]  # TODO: In Shun's proposed model, O takes 4 parameters - s (current state) is also required.
            transition_prob = 0
            for s_index, s in enumerate(S):
                transition_prob += T[s][a][s_prime] * b[s_index]
            pr_z += new_observation * transition_prob
        print("pr_z({} | {}, {}) = {}".format(z, b, a, pr_z))
        return pr_z

    @memoize
    def tau(self, belief_state, action: str, observation: str, O: dict, T: dict, S: list):
        r"""Implement the tau function for computing belief update.

        Implements the function:

        .. math::
            \tau(b, a, z)(s') = \frac{1}{Pr(z | b, a)} * O(s', a, z)\sum_{s \in S} T(s, a, s')b(s)

        We won't return a specific belief point, but the entire belief state:

        .. math::
            \tau(b, a, z_t)

        So, we return a list indexed by states in `S`.

        Args:
            node (Node): the node containing belief state b
            action (str): the action a
            observation (str): the observation z
            O (dict[str, dict[str, dict[str, float]]]): the observation probability function. Function of state, action, observation.
            T (dict[str, dict[str, dict[str, float]]]): the transition probability function. Function of state, action, state.
            S (list[str]): list of states
        Returns:
            list[float]: the updated belief state, given b, a, and z.
        """
        z = observation
        """:type: str"""
        b = belief_state
        """:type: list[float]"""
        a = action
        """:type: str"""

        new_belief_state = []
        for s_prime in S:
            #calculate Pr(z | b,a)
            pr_z = self.pr_z(observation=z, belief_state=b, action=a, O=O, T=T, S=S)

            #calculate tau
            normalizing_factor = 1.0/pr_z
            temp = normalizing_factor * float(O[observation][(a, s_prime)])#[s_prime][a][observation]
            """\frac{1}{Pr(z | b, a)} * O(s', a, z)"""

            transition_sum = 0
            """\sum_{s \in S} T(s, a, s')b(s)"""

            for s_index, s in enumerate(S):
                transition_sum += T[s][a][s_prime] * b[s_index]

            new_belief_state.append(temp * transition_sum)

        print("tau({},{},{}) = {}".format(b,a,z,new_belief_state))

        return new_belief_state


class Node(object):
    """Represent a Node in an AO_Tree.

    Represent, in a general sense, a node of an AND/OR Tree. An AO_Tree is composed of Nodes.

    Args:
        parent (Node): initial parent of this Node.
        children (list[Node]): initial list of children of this Node
        upper_bound (float): initial upper bound of this Node. upper_bound is updated when a new node is expanded in the AO_Tree.
        lower_bound (float): initial lower bound of this Node. lower_bound is updated when a new node is expanded in the AO_Tree.

    Attributes:
        parent (Node): the parent Node of this Node.
        children (list[Node]): children of this Node.
        upper_bound (float): Upper bound for this Node. May be updated when a new node is expanded in the AO_Tree.
        lower_bound (float): Lower bound of this Node. May be updated when a new node is expanded in the AO_Tree.
        initial_upper_bound (float): The initial upper bound. Will not be changed when this node is expanded.
        initial_lower_bound (float): The initial lower bound. Will not be changed when this node is expanded.
        depth (int): current depth (measured from root) of this Node.

    """

    def __init__(self, parent=None, children: list=(), upper_bound: float=0.0, lower_bound: float=0.0, belief_state=()):
        self.parent = parent
        self.children = list(children)
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.depth = 0
        self.initial_lower_bound = lower_bound
        self.initial_upper_bound = upper_bound
        self.belief_state = list(belief_state)

        self._update_depth()

        if self.parent is not None:
            self.add_as_child(self.parent)

    def add_as_child(self, node):
        """Add this Node as a child to parent Node `node`.

        This function removes this Node from its `parent`'s `children` list (if there is a parent) before setting this Node's `parent` to `node`.

        Args:
            node (Node): the new parent of this Node.

        """
        if self.parent is not None and self.parent is not node:
            self.parent.children.remove(self)

        self.parent = node
        """:type : Node"""

        if self.parent is not None:
            self.parent.children.append(self)

        self._update_depth()

    def _update_depth(self):
        """Abstract function hook called by add_as_child.
        Should be implemented by subclasses to decide what their depth should be when added as a child.
        """
        pass

    def size(self):
        """Calculate the memory this Node is using.

        :return:
        """
        size = 0
        size += 8 + len(self.children)*8 + sys.getsizeof(self.upper_bound) + sys.getsizeof(self.lower_bound) + sys.getsizeof(self.depth)
        size += self._size()  # hook for subclasses
        return size

    def _size(self):
        """Hook for subclasses to account for size of additional attributes.

        :return:
        """
        return 0


class OR_Node(Node):
    """Subclass Node to include belief state.

    OR_Nodes include a belief state.

    Args:
        belief_state (list[float]): the belief state of this node. len(`belief_state`) must be the same as the number of states in the POMDP.
        reach_probability (float): the probability that the `belief_state` of this OR_Node is reached from the initial belief state.
        obs_prob (float): the probability of the observation that gave rise to this OR_Node.

    Attributes:
        action Optional(str): the action that gave rise to this OR_Node (i.e. parent.action). `action` is None for the root OR_Node.
    """

    def __init__(self, parent: Node=None, children: list=(), upper_bound: float=0.0, lower_bound: float=0.0,
                 belief_state: list=(), reach_probability: float=0.0, obs_prob: float=0.0, observation=None, action=None):
        super().__init__(parent, children, upper_bound, lower_bound, belief_state)
        self.reach_probability = reach_probability
        self.obs_prob = obs_prob
        self._value = 0  # the expand function of Tree will need to update value, since it depends on V.
        self.observation = observation

    def value(self):
        return self._value

    def _size(self):
        return sum([sys.getsizeof(elem) for elem in self.belief_state])

    def _update_depth(self):
        """Implement _update_depth as `parent.depth` +1.

        The depth of an OR Node is one more than its parent.

        """
        if self.parent is not None:
            self.depth = self.parent.depth + 1

    def __str__(self):
        s = ['\nOR_Node.']
        s.append('Belief state: {}'.format(self.belief_state))
        if self.observation is not None:
            s.append('Observation: {}'.format(self.observation))
        s.append('Reach probability: {}'.format(self.reach_probability))
        s.append('observation probability: {}'.format(self.obs_prob))
        s.append('Value: {}'.format(self.value()))
        s.append('Upper Bound: {}'.format(self.upper_bound))
        s.append('Lower Bound: {}'.format(self.lower_bound))
        s.append('Parent: {}'.format(self.parent))
        s.append('\n')
        return '\n'.join(s)



class AND_Node(Node):
    """Subclass Node to include action choice.

    AND_Nodes include an action choice.

    Args:
        action (str): the action choice for this AND_Node.

    """

    def __init__(self, parent=None, children: list=(), upper_bound: float=0.0, lower_bound: float=0.0, action: str=None, belief_state=()):
        super().__init__(parent, children, upper_bound, lower_bound, belief_state)
        self.action = action
        self.successors = []

    def value(self):
        return sum([child.value() * child.obs_prob for child in self.children])

    def _size(self):
        return sys.getsizeof(self.action)

    def _update_depth(self):
        """Implement _update_depth as `parent.depth` + 0.

        The depth of an AND Node is the same as its parent.


        """
        if self.parent is not None:
            self.depth = self.parent.depth

    def __str__(self):
        s = ['\nAND_Node.']
        s.append('Belief state: {}'.format(self.belief_state))
        s.append('Action: {}'.format(self.action))
        s.append('Value: {}'.format(self.value()))
        s.append('Upper Bound: {}'.format(self.upper_bound))
        s.append('Lower Bound: {}'.format(self.lower_bound))
        s.append('Parent: {}'.format(self.parent))
        s.append('\n')
        return '\n'.join(s)
