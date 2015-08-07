from Heuristics.Bounds import LowerBound, UpperBound, Bound

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

    def __init__(self, root, upper_bound_function: UpperBound, lower_bound_function: LowerBound, actions: list=[], observations: list=[]):
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

    def expand(self, node, O: dict, T: dict, S: list, R_B):
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
        """

        #First remove node from fringe.
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

        for action in self.actions:
            new_child = AND_Node(parent=node, action=action)
            and_children.append(new_child)

        # for each new AND_Node, create an OR_Node and compute its belief state according to:
        # b_t(s') = \tau(b_{t-1}, a_{t-1}, z_t)(s')
        for and_child in and_children: #iterate over all new AND_Nodes.
            a = and_child.action
            for observation in self.observations:  # this generates each child of the and_child
                new_belief_state = self.tau(node=node, action=a, observation=observation, O=O, T=T, S=S)
                obs_prob = self.pr_z(observation=observation, belief_state=node.belief_state, action=a,
                                     O=O, T=T, S=S)

                # probability of reaching new_child is the probability of reaching node * Pr(z | b,a)
                reach_probability = node.reach_probability*obs_prob

                new_child = OR_Node(parent=and_child, belief_state=new_belief_state,obs_prob=obs_prob,
                                    reach_probability=reach_probability, lower_bound=self.L.value(new_belief_state),
                                    upper_bound=self.U.value(new_belief_state))
                self.fringe.append(new_child)
                self.fringe_beliefs.append(new_child.belief_state)

                #TODO: Should we propagate after each child is created, or after they are all created? I think it doesn't matter
                self.propagate(R_B, O, T, S, fringe_nodes=self.fringe)

    def propagate(self, R_B, O, T, S, fringe_nodes: list=[]):
        r"""Propagate upper and lower bound information to root.

        Updates upper and lower bounds for ancestors of nodes in `fringe_nodes`.

        And we propagate the upper bound as:


        In both cases, F(T) is the set of fringe nodes in tree T, U_T(b) and L_T(b) represent upper and lower bounds on V'(b) associated to belief state b,
        U_T(b,a) and L_T(b,a) represent corresponding bounds on Q*(b,a), and L(b) and U(b) are the bounds of the fringe nodes.



        Args:
            R_B (dict[tuple[list[float], str], float]): The payoff function, mapping a belief state and action pair to a reward.
            fringe_nodes Optional(list[OR_Node]): the nodes on the fringe from which to start bound propagation.
        """
        for fringe_node in fringe_nodes:
            node_ptr = fringe_node.parent
            """:type: Node"""
            while node_ptr is not None:
                self.update_lower_bound(node_ptr, R_B, O, T, S)
                self.update_upper_bound(node_ptr)
                node_ptr = node_ptr.parent

    def update_lower_bound(self, node, R_B, O, T, S):
        r"""Update the lower bound of `node`.

        We propagate the lower bounds of the fringe nodes up the tree with these equations:

        .. math::
            L_T(b) = \begin{cases} L(b), & \text{if } b \in F(T) \\ \text{max}_{a \in A} L_T(b,a), & \text{otherwise}\end{cases}

            L_T(b,a) = R_B(b,a) + \gamma \sum_{z \in Z} \text{Pr}(z | b,a)L_T(\tau(b,a,z))


        Args:
            node (Node): the node whose lower bound should be updated
            R_B (dict[tuple[list[float], str], float]): the payoff function, mapping a belief state and action pair to a reward.
        """
        node.lower_bound = self.bound_t(node.belief_state, R_B, O, T, S, self.L)

    def update_upper_bound(self, node, R_B, O, T, S):
        r"""Update the upper bound of `node`.

        We propagate the upper bounds of the fringe nodes up the tree with these equations:
        .. math::
            U_T(b) = \begin{cases}U(b), & \text{if } b \in F(T) \\ \text{max}_{a \in A} U_T(b,a), & \text{otherwise}\end{cases}

            U_T(b,a) = R_B(b,a) + \gamma \sum_{z \in Z} \text{Pr}(z | b,a)U_T(\tau(b,a,z))
        :param node:
        :param R_B:
        :param O:
        :param T:
        :param S:
        :return:
        """
        node.upper_bound = self.bound_t(node.belief_state, R_B, O, T, S, self.U)

    def bound_t(self, belief_state, R_B, O, T, S, fringe_function: Bound):
        """Abstract update to upper bound and lower bound.

        By taking a function parameter, abstract the calculation for bound propagation.

        :param belief_state:
        :param R_B:
        :param O:
        :param T:
        :param S:
        :param fringe_function:
        :param action_function:
        :return:
        """
        if belief_state in self.fringe_beliefs:
            return fringe_function.value(belief_state)
        else:
            new_lower_bound = 0
            for action in self.actions:
                bound_t = self.bound_action(belief_state=belief_state,
                                            R_B=R_B, O=O, T=T, S=S, action=action, fringe_function=fringe_function)
                if bound_t > new_lower_bound:
                    new_lower_bound = bound_t
            return new_lower_bound

    def bound_action(self, belief_state, R_B, O, T, S, action, fringe_function):
        r"""
        Calculate the sub-function used by bound_t.

        Implement:
        .. math::
            U_T(b,a) = R_B(b,a) + \gamma \sum_{z \in Z} \text{Pr}(z | b,a)U_T(\tau(b,a,z))
        :param belief_state:
        :param R_B:
        :param O:
        :param T:
        :param S:
        :param action:
        :param fringe_function:
        :return:
        """
        weighted_lower_bound = 0
        for observation in self.observations:
            weighted_lower_bound += (self.pr_z(observation=observation, belief_state=belief_state,
                                               action=action, O=O, S=S, T=T) *
                                     self.bound_t(self.tau(belief_state, action=action,
                                                           observation=observation, O=O, T=T, S=S),
                                                  R_B, O, T, S, fringe_function=fringe_function)
                                     )
        return R_B[(belief_state, action)] + weighted_lower_bound

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
            new_observation = O[s_prime][a][z]  # TODO: In Shun's proposed model, O takes 4 parameters - s (current state) is also required.
            transition_prob = 0
            for s_index, s in enumerate(S):
                transition_prob += T[s][a][s_prime] * b[s_index]
            pr_z += new_observation * transition_prob

        return pr_z

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
            temp = normalizing_factor * O[s_prime][a][observation]
            """\frac{1}{Pr(z | b, a)} * O(s', a, z)"""

            transition_sum = 0
            """\sum_{s \in S} T(s, a, s')b(s)"""

            for s_index, s in enumerate(S):
                transition_sum += T[s][a][s_prime] * b[s_index]

            new_belief_state.append(temp * transition_sum)

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
        depth (int): current depth (measured from root) of this Node.

    """

    def __init__(self, parent=None, children: list=[], upper_bound: float=0.0, lower_bound: float=0.0):
        self.parent = parent
        self.children = children
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.depth = 0

        self._update_depth()

        if self.parent is not None:
            self.add_as_child(self.parent)

    def add_as_child(self, node):
        """Add this Node as a child to parent Node `node`.

        This function removes this Node from its `parent`'s `children` list (if there is a parent) before setting this Node's `parent` to `node`.

        Args:
            node (Node): the new parent of this Node.

        """
        if self.parent is not None:
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
    """

    def __init__(self, parent: Node=None, children: list=[], upper_bound: float=0.0, lower_bound: float=0.0,
                 belief_state: list=[], reach_probability: float=0.0, obs_prob: float=0.0):
        super(Node, self).__init__(parent, children, upper_bound, lower_bound)
        self.belief_state = belief_state
        self.reach_probability = reach_probability
        self.obs_prob = obs_prob
        self.value = 0  # the expand function of Tree will need to update value, since it depends on V.

    def value(self):
        return self.value

    def _size(self):
        return sum([sys.getsizeof(elem) for elem in self.belief_state])

    def _update_depth(self,):
        """Implement _update_depth as `parent.depth` +1.

        The depth of an OR Node is one more than its parent.

        """
        if self.parent is not None:
            self.depth = self.parent.depth + 1


class AND_Node(Node):
    """Subclass Node to include action choice.

    AND_Nodes include an action choice.

    Args:
        action (str): the action choice for this AND_Node.

    """

    def __init__(self, parent=None, children: list=[], upper_bound: float=0.0, lower_bound: float=0.0, action: str=None):
        super(Node, self).__init__(parent, children, upper_bound, lower_bound)
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
