class EnumerativeMDP(object):
    def __init__(self, states, reward_function, transition_function, initial_states, goal_states):
        __validate_states(states)
        __validate_reward_function(reward_function, states)

        transition_function = __fix_missings_in_transition_function(transition_function, states)
        __validate_transition_function(transition_function, states)

        __validate_initial_states(initial_states, states)
        __validate_goal_states(goal_states, states)

        self.states = states
        self.reward_function = reward_function
        self.transition_function = transition_function
        self.initial_states = initial_states
        self.goal_states = goal_states

    def __validate_states(self, states):
        if len(states) == 0: raise ValueError("state array should have at least one state")


    def __validate_reward_function(self, reward_function, states):
        if len(reward_function) != len(states): raise ValueError("The reward function must be have a value for each state.")

        for state in reward_function.keys():
            if state not in states: raise ValueError("Invalid state [%s] defined in reward fuction."%state)

    def __fix_missings_in_transition_function(self, transition_function, states):
        actions = transition_function.keys()

        for action_name in actions:
            for from_state in transition_function[action_name].keys():
                for to_state in states:
                    if not transition_function[action_name][from_state].has_key(to_state):
                        transition_function[action_name][from_state][to_state] = 0.0

        return transition_function

    def __validate_transition_function(self, transition_function, states):
        # invalid state transition definition in transition_function
        for action_name in transition_function.keys():
            if len(states) != len(transition_function[action_name]):
                raise ValueError("action [%s] must define the transition probabilities for all states in transition function."%action_name)

            for from_state in transition_function[action_name].keys():
                if from_state not in states:
                    raise ValueError("Unrecognized origin state [%s] for action [%s] in transition function."%(from_state, action_name))

                for to_state in transition_function[action_name][from_state].keys():
                    if to_state not in states:
                        raise ValueError("Unrecognized destination state [%s] from [%s] transition for action [%s] in transition function."%(to_state, from_state, action_name))

            # invalid transition probability in action_probabilities
            for action_name in transition_function.keys():
                for from_state in transition_function[action_name].keys():
                    probability = sum(transition_function[action_name][from_state].values())

                    if probability != 1.0:
                        raise ValueError("Invalid probability distribution on [%s] transition in action [%s]. The sum of all transitions in this state must be 1 (one)."%(from_state, action_name))

    def __validate_initial_states(self, initial_states, states):
        for initial_state in initial_states:
            if initial_state not in states:
                raise ValueError("Unrecognized state [%s] in initial states."%initial_state)

    def __validate_goal_states(self, goal_states, states):
        for goal_state in goal_states:
            if goal_state not in states:
                raise ValueError("Unrecognized state [%s] in goal states."%goal_state)