__author__ = 'Victor Szczepanski'


class AO_Tree(object):

    def __init__(self):
        pass


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

    """

    def __init__(self, parent: AO_Tree.Node=None, children: list=[], upper_bound: float=0.0, lower_bound: float=0.0):
        self.parent = parent
        self.children = children
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound


class OR_Node(Node):

    def value(self):
        pass


class AND_Node(Node):

    def value(self):
        pass
