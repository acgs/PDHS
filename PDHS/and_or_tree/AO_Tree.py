__author__ = 'Victor Szczepanski'


class Tree(object):
    """Represent an AND/OR Tree.

    Args:
        root (OR_Node): the root node of this AND/OR Tree.
        actions (list[str]): the list of actions we can expand from an OR_Node.
        observations (list[str]): the list of observations we can expand from an AND_Node.

    Attributes:
        root (OR_Node): the root node of this AND/OR Tree.
        actions (list[str]): the list of actions we can expand from an OR_Node.
        observations (list[str]): the list of observations we can expand from an AND_Node.
        fringe (list[OR_Node]): The fringe nodes that can be chosen for expansion.
    """

    def __init__(self, root, actions: list=[], observations: list=[]):
        self.root = root
        self.actions = actions
        self.observations = observations
        self.fringe = [self.root]

    def expand(self, node, O: dict, T: dict, S: list):
        """Expand a given node in this tree.

        Expands a node as described in "Online Planning Algorithms for POMDPs" by Ross, Pineau, et. al, Journal of Artifical Intelligence Research 32, (2008), 663-704.

        In particular, implements the function

        .. math::
            b_t(s') = \\tau(b_{t-1}, a_{t-1}, z_t)(s') = \\frac{1}{Pr(z_t l b_{t-1}, a_{t-1})} * O(s', a_{t-1}, z_t) \Sum_{s \in S} T(s, a_{t-1}, s')b_{t-1}(s)

            Pr(z | b, a) = \Sum_{s' \in S} O(s', a, z) \Sum_{s \in S} T(s, a, s')*b(s)

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

        #Generate AND_Nodes for every action
        and_children = []
        """:type : list[AND_Node]"""

        for action in self.actions:
            new_child = AND_Node(parent=node, action=action)
            and_children.append(new_child)

        # for each new AND_Node, create an OR_Node and compute its belief state according to:
        # b_t(s') = \tau(b_{t-1}, a_{t-1}, z_t)(s') = \frac{1}{Pr(z_t | b_{t-1}, a_{t-1})} * O(s', a_{t-1}, z_t) \Sum_{s \in S} T(s, a_{t-1}, s')b_{t-1}(s)
        for and_child in and_children:
            for observation in self.observations:
                #compute belief state
                belief_state = 0
                normalizing_factor = 0
                for s_prime in S:
                    new_observation = O[s_prime][action][observation]
                    transition_prob = 0
                    for s in S:
                        transition_prob += T[s][action][s_prime] * node.belief_state
                    normalizing_factor += new_observation * transition_prob

                normalizing_factor = 1/normalizing_factor

                new_child = OR_Node(parent=and_child, belief_state=belief_state)


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
        self.parent.children.append(self)

        self._update_depth()

    def _update_depth(self):
        """Abstract function hook called by add_as_child.
        Should be implemented by subclasses to decide what their depth should be when added as a child.
        """
        pass


class OR_Node(Node):
    """Subclass Node to include belief state.

    OR_Nodes include a belief state.

    Args:
        belief_state (list[float]): the belief state of this node. len(`belief_state`) must be the same as the number of states in the POMDP.

    """

    def __init__(self, parent: Node=None, children: list=[], upper_bound: float=0.0, lower_bound: float=0.0, belief_state: list=[]):
        super(Node, self).__init__(parent, children, upper_bound, lower_bound)
        self.belief_state = belief_state

    def value(self):
        pass

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

    def value(self):
        pass

    def _update_depth(self):
        """Implement _update_depth as `parent.depth` + 0.

        The depth of an AND Node is the same as its parent.


        """
        if self.parent is not None:
            self.depth = self.parent.depth
