from Models.models import Player, POMDPModel, ValueFunction

__author__ = 'Victor Szczepanski'


class Bound(object):
    """Generically describe a 'bound'. Should be inherited from by upper and lower bounds."""

    def value(self, belief_state: list):
        """Value of this bound at belief state `belief_state`.

        All bounds must be able to calculate their value at a belief state. However, this function is abstract in `Bound`.

        Args:
            belief_state (list[float]): the belief state to compute the value for.
        Returns:
            float: the value of this bound at `belief_state`.
        """
        raise NotImplementedError("This is an instance of an abstract Bound and should not be instantiated.\n"
                                  "You probably meant to use a subclass that implements value.")


class UpperBound(Bound):
    """Generically describes an 'upper bound'. As with Bound, this is an abstract class."""


class LowerBound(Bound):
    """Generically describes a 'lower bound'. As with Bound, this is an abstract class."""


class LBValueFunction(LowerBound, ValueFunction):
    """A POMDP Value Function as a lower bound.
    """

    def value(self, belief_state: list):
        r"""Override value and return the value of a belief state as the maximum value returned by one of the alpha-vectors for this belief state.

        In particular, implement the function:

        .. math::
            V(b) = max{\alpha \in alpha_vectors} \sum_{s \in S} \alpha(s) * b(s)

        Args:
            belief_state (list[float]): a belief state, indexed by state index.

        Returns:
            float: the value corresponding to the best alpha vector for `belief_state`

        """
        value = 0.0
        for alpha in self.alpha_vectors:
            value_sum = 0
            for s_index in range(len(self.states)):
                value_sum += alpha[s_index] * belief_state[s_index]
            if value_sum > value:
                value = value_sum

        return value

