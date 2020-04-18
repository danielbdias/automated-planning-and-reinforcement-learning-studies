"""Module with classes to support a structure that represents
   a factored Markov Decision Process in Probabilistic Planning."""

from ..helpers import validate_defined_argument

def build_state_variables_set(state_variable_indentifiers, state_variables_set_name, base_state_variables_set=None):
    """Build and validate a state set."""

    validate_defined_argument(state_variable_indentifiers, state_variables_set_name)

    if len(state_variable_indentifiers) == 0:
        raise ValueError(f"The {state_variables_set_name} should have at least one state variable")

    state_variables_as_set = set(state_variable_indentifiers)
    if len(state_variables_as_set) != len(state_variable_indentifiers):
        raise ValueError(f"There is a repeated state variable identifier in the {state_variables_set_name}")

    if base_state_variables_set is not None:
        for state_variable in state_variable_indentifiers:
            if state_variable not in base_state_variables_set:
                raise ValueError(f"Unrecognized state variable [{state_variable}] in {state_variables_set_name}")

    return state_variables_as_set

class FactoredMDP:
    """Represents a factored Markov Decision Process.

    Attributes:
        state_variables (set): variables used to model a state in this MDP representation
        reward_function (pyddlib.add.ADD): algebraic decision diagram representing reward given the states variables representation
        actions (set): actions that are modelled in this MDP
        transition_function (`FactoredTransitionFunction`): an object that given an origin state variable, an action and a destination state variable returns the transition
                                                            probability from origin state variable to destination state variable
        initial_states (set): (optional) initial factored states modelled in this MDP
        goal_states (set): (optional) goal factored states modelled in this MDP
    """

    def __init__(self, state_variables, reward_function, transition_function,
                 initial_states=None, goal_states=None):
        """Initializes a new representation of an enumerative Markov Decision Process.

        Parameters:
            state_variables (list): named state variables for this MDP
            reward_function (pyddlib.add.ADD): ADD that maps a state variable configuration to the respective numeric reward
            transition_function (dict): maps an action string to another dict that maps a state variable to an ADD defining
                                        transition probabilities with respect to other state variables
            initial_states (list): list of initial factored states for this MDP
            goal_states (list): list of initial factored states for this MDP

        Returns:
            instance (FactoredMDP): a factored Markov Decision Process
        """
        self.state_variables = build_state_variables_set(state_variables, "state variables")
        # self.reward_function = build_reward_function(reward_function, self.state_variables)
        # self.transition_function = build_transition_funtion(transition_function, self.state_variables)
        # self.actions = build_action_set(self.transition_function.actions)

        # if initial_states:
        #     self.initial_states = build_state_set(initial_states, "initial states", self.state_variables)
        # else:
        #     self.initial_states = set()

        # if goal_states:
        #     self.goal_states = build_state_set(goal_states, "goal states", self.state_variables)
        # else:
        #     self.goal_states = set()

    def reward(self, state):
        return self.reward_function[state]

    def transition(self, from_state, action, to_state):
        return self.transition_function.get_transition_probability(from_state, action, to_state)
