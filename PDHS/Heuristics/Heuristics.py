__author__ = 'Victor Szczepanski'

class Heuristic(object):
    """Define abstract Heuristic class.

    A Heuristic is an interface that exposes a single function: select.

    Subclasses should implement select.
    """
    def __init__(self):
        pass

    def select(self, nodes: list):
        """Select a `Node` from `nodes` based on this Heuristic.

        This function should be implemented by subclasses.
        Args:
            nodes (list[Node]): the nodes to select from.
        Returns:
            Node: the best Node based on this heuristic.
        """
        raise NotImplementedError("Heuristic should not be used directly, or subclass does not implement select.")

class HansenHeuristic(Heuristic):
    """Implement the Heuristic described by Hansen (1997).

    We require one additional piece of information not contained in the nodes: beta

    Attributes:
        beta (float): the ``discount factor`` of the POMDP.

    """

    def __init__(self, beta: float=0.0):
        self.beta = float(beta)

    def select(self, nodes: list):
        r"""Select a `Node` from `nodes` using Hansen's Heuristic

        Implements the function

        .. math::
            (UB(b) - V(b)) * (REACHPROB(b) * \beta^{DEPTH(b)})

        Args:
            nodes (list[OR_Node]): the nodes to select from.
        Returns:
            OR_Node: The node whose belief state maximizes the above function.
        """
        selected_node = nodes[0]
        print("In Hansen select. nodes: {}, initial selected node: {}".format(nodes, selected_node))
        """:type : OR_Node"""
        selected_value = ((selected_node.upper_bound - selected_node.value()) *
                          (selected_node.reach_probability * self.beta**selected_node.depth)
                          )
        for node in nodes:
            test_value = (node.upper_bound - node.value()) * (node.reach_probability * self.beta**node.depth)
            if test_value > selected_value:
                selected_value = test_value
                selected_node = node

        return selected_node

