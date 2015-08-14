import copy
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
            for s_index in range(len(self.pomdp_states)):
                value_sum += alpha[s_index] * belief_state[s_index]
            if value_sum > value:
                value = value_sum

        return value



class ImprovableUpperBound(UpperBound):
    """
    Improvable upper bounds implement an ``improve`` function.
    """

    def __init__(self):
        self.is_converged = False
        self.improve_iterations = 0

    def improve(self):
        raise NotImplementedError("This Base class should be subclassed to be used.")


class UB_MDP(ImprovableUpperBound):
    r"""Implement the upper bound as the value function of the underlying MDP for a given POMDP.

    We use equation:

    .. math::
        V_{t+1}^{MDP}(s) = max_{a \in A} \left[ R(s,a) + \gamma \sum_{s' \in S} T(s,a,s')V_t^{MDP}(s') \right]

    to improve the upper bound.

    Then the value for a POMDP is compututed as:

    .. math::
        \hat{V}(b) = \sum_{s \in S} V^{MDP}(s)b(s)

    From Littman et al., 1995.

    Args:
        S (list[str]): The list of states for the underlying MDP.
        A (list[str]): The actions that can be taken by an agent in the MDP/POMDP.
        T (dict[str, dict[str, dict[str, float]]]): Transition probability - mapping of state to action to state to probability that represents the probability that the next state is θ^t+1 when the current state is θ^t and the action is a_1.
        initial_V Optional(list[float]): The initial upper bound to use - len(`S`) == len(`initial_V`)

    Attributes:
        V_MDP (list[float]): the value function for the underlying MDP - maps each state to a value.
        T (dict[str, dict[str, dict[str, float]]]): Transition probability - mapping of state to action to state to probability that represents the probability that the next state is θ^t+1 when the current state is θ^t and the action is a_1.

        converged (bool): True if this value function has converged (maximum update is less than

    """
    def __init__(self, S, A, T, R, gamma, initial_V: list=[]):
        super().__init__()
        self.S = S
        self.A = A
        self.T = T
        self.R = R
        self.gamma = gamma
        self.V_MDP = initial_V
        if len(self.V_MDP) is 0:
            # populate V_MDP with initial values for each state
            for i in range(len(S)):
                self.V_MDP.append(0.0)
        assert len(self.S) is len(self.V_MDP)
        
    def improve(self):
        """
        Improve this upper bound by one iteration
        :return:
        """
        V_MDP_prime = copy.copy(self.V_MDP)
        assert(V_MDP_prime is not self.V_MDP and V_MDP_prime == self.V_MDP)
        for s, state in enumerate(self.S):
            #find best value
            best = 0
            for a in self.A:
                value = float(self.R[(a, state)])
                discounted_sum = 0
                for s_prime_index, s_prime in enumerate(self.S):
                    discounted_sum += self.T[state][a][s_prime] * self.V_MDP[s_prime_index]
                value = value + self.gamma * discounted_sum
                if value > best:
                    best = value
            V_MDP_prime[s] = best
        self.improve_iterations += 1
        # determine if we have reached convergence.
        self.V_MDP = V_MDP_prime

    def value(self, belief_state: list):
        r"""Implement belief state lookup.

        Implement function:
        .. math::
            \hat{V}(b) = \sum_{s \in S} V^{MDP}(s)b(s)

        Args:
            belief_state (list[float]): the belief state to calculate the value of.
        Returns:
            float: The value of this upper bound at `belief_state`
        """
        return sum([self.V_MDP[s]*belief_state[s] for s in range(len(self.S))])


class UB_QMDP(ImprovableUpperBound):
    r"""Implement the upper bound as a value function that considers partial obervability as disappearing after one step.

    Implement function:

    .. math::
        Q_{t+1}^{MDP}(s,a) = R(s,a) + \gamma \sum_{s' \in S} T(s,a,s')V_t^{MDP}(s')

    From Littman et al., 1995.

    Note:
        `initial_V_MDP` should not be improved prior to being passed to a new UB_QMDP, or improved afterwards, since this UB_QMDP will improve it.
        `initial_V_MDP`'s value function may be called independently, however.

    Attributes:
        S (list[str]): The states of the POMDP
        A (list[str]): The actions of the POMDP
        T (dict[str, dict[str, dict[str, float]]]): Transition probability - mapping of state to action to state to probability that represents the probability that the next state is θ^t+1 when the current state is θ^t and the action is a_1.
        R (Dict[tuple[ACTION,STATE], float]): Mapping from action/state tuple to a payoff real value. Represents the immediate payoff of taking an action in a state.
        initial_V_MDP (UB_MDP): The value function for the underlying MDP of a POMDP.
        initial_Q_MDP (dict[str, list[float]]): Mapping of an action to a list of values, indexed by state index.

    """

    def __init__(self, S, A, T, R, gamma, initial_V_MDP, initial_Q_MDP={}):
        super().__init__()
        self.S = S
        self.A = A
        self.T = T
        self.R = R
        self.gamma = gamma
        self.V_MDP = initial_V_MDP
        self.Q_MDP = initial_Q_MDP
        if len(self.Q_MDP) is 0:
            for a in self.A:
                self.Q_MDP[a] = []
                for s in self.S:
                    self.Q_MDP[a].append(0)

    def improve(self):
        for s_index, s in enumerate(self.S):
            for a in self.A:
                value = float(self.R[(a, s)])
                value_sum = 0
                for s_prime_index, s_prime in enumerate(self.S):
                    value_sum += self.T[s][a][s_prime] * self.V_MDP.V_MDP[s_prime_index]
                self.Q_MDP[a][s_index] = value + self.gamma * value_sum

        #Then we improve V_t^MDP for the next step
        self.improve_iterations += 1
        self.V_MDP.improve()

    def value(self, belief_state: list):
        r"""

        Implement function:

        .. math::
            V_t(b) = max_{\alpha \in \Gamma_t} \sum_{s \in S} \alpha(s)*b(s)

        Note:
            :math:`\Gamma_t` for QMDP contains one :math:`\alpha`-vector :math:`\alpha^a(s) = Q_t^{MDP}(s,a)` for each :math:`a \in A`.

        Args:
            belief_state (list[float]): The belief state to calculate the value for.

        Returns:
            float: the value corresponding to `belief_state` for this QMDP.
        """
        value = 0.0
        for a in self.A:
            value_sum = 0
            for s_index in range(len(self.S)):
                value_sum += self.Q_MDP[a][s_index] * belief_state[s_index]
            if value_sum > value:
                value = value_sum

        return value


class UB_FIB(ImprovableUpperBound):
    r"""Implement the Fast Informed Bound.

    Implement function:

    .. math::
        \alpha_{t+1}^a(s) = R(s,a) + \gamma \sum_{z \in Z} max_{\alpha_t \in \Gamma_t} \sum_{s' \in S} O(s', a, z)T(s,a,s')\alpha_t(s').

    Args:
        converged_QMDP (dict[str, list[float]]): the alpha vectors, one for each action, generated by QMDP at convergance.

    Attributes:
        alphas (dict[str, list[float]]: the current alpha vectors, defined for each action and mapped to a list of values for each state, indexed by state index.


    """

    def __init__(self, R, gamma, S, A, O, T, Z, converged_QMDP):
        super().__init__()
        self.alphas = converged_QMDP
        self.R = R
        self.gamma = gamma
        self.S = S
        self.A = A
        self.O = O
        self.T = T
        self.Z = Z

    def improve(self):
        new_alphas = copy.copy(self.alphas)
        for a in self.A:
            for s_index, s in enumerate(self.S):
                z_sum = 0
                for z in self.Z:
                    max_alpha_value = 0
                    inner_sum = 0
                    for action, alpha in self.alphas.items():
                        inner_sum = sum([float(self.O[z][(a, s_prime)]) * float(self.T[s][a][s_prime])
                                         * alpha[s_prime_index] for s_prime_index, s_prime in enumerate(self.S)])
                        if inner_sum > max_alpha_value:
                            max_alpha_value = inner_sum
                    z_sum += max_alpha_value
                new_alphas[a][s_index] = float(self.R[(a, s)]) + self.gamma * z_sum

        self.improve_iterations += 1
        self.alphas = new_alphas

    def value(self, belief_state: list):
        r"""

        Implement function:

        .. math::
            V_t(b) = max_{\alpha \in \Gamma_t} \sum_{s \in S} \alpha(s)*b(s)

        Note:
            :math:`\Gamma_t` for QMDP contains one :math:`\alpha`-vector :math:`\alpha^a(s) = Q_t^{MDP}(s,a)` for each :math:`a \in A`.

        Args:
            belief_state (list[float]): The belief state to calculate the value for.

        Returns:
            float: the value corresponding to `belief_state` for this QMDP.
        """
        value = 0.0
        for a in self.A:
            value_sum = 0
            for s_index in range(len(self.S)):
                value_sum += self.alphas[a][s_index] * belief_state[s_index]
            if value_sum > value:
                value = value_sum

        return value
