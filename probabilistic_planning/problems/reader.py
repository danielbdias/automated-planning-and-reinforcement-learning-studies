from probabilistic_planning.structures.mdp import EnumerativeMDP

def clean_string(value, remove_tabs = False, remove_spaces = False, remove_line_breaks = False):
    if remove_tabs:
        value = value.replace("\t", "")

    if remove_spaces:
        value = value.replace(" ", "")

    if remove_line_breaks:
        value = value.replace("\r", "").replace("\n", "")

    return value


def read_state_section(file):
    line = file.readline()
    line = clean_string(line, remove_tabs = True, remove_spaces = True, remove_line_breaks = True)

    states = line.split(",")

    while True:
        line = file.readline()

        if not line:
            raise Exception("endstates token not found !")

        if line.startswith("endstates"):
            return states


def read_action_section(file, line):
    transition_probabilities = {}

    data = line.split(" ")
    action_name = clean_string(data[1], remove_line_breaks = True)

    while True:
        line = file.readline()

        if not line:
            raise Exception("endaction token not found !")

        if line.startswith("endaction"):
            return action_name, transition_probabilities

        line = clean_string(line, remove_tabs = True)
        data = line.split(" ")

        from_state = data[0]
        to_state = data[1]
        probability = float(data[2])

        if from_state not in transition_probabilities.keys():
            transition_probabilities[from_state] = {}

        transition_probabilities[from_state][to_state] = probability


def read_reward_section(file):
    reward = {}

    while True:
        line = file.readline()

        if not line:
            raise Exception("endreward token not found !")

        if line.startswith("endreward"):
            return reward

        line = clean_string(line, remove_tabs = True, remove_line_breaks = True)
        data = line.split(" ")

        state = data[0]
        value = float(data[1])

        reward[state] = value


def read_initial_state_section(file):
    initial_states = []

    while True:
        line = file.readline()

        if not line:
            raise Exception("endinitialstate token not found !")

        if line.startswith("endinitialstate"):
            return initial_states

        state = clean_string(line, remove_tabs = True)
        initial_states.append(state)


def read_goal_state_section(file):
    goal_states = []

    while True:
        line = file.readline()

        if not line:
            raise Exception("endgoalstate token not found !")

        if line.startswith("endgoalstate"):
            return goal_states

        state = clean_string(line, remove_tabs = True)
        goal_states.append(state)


def read_problem_file(problem_file):
    states = None
    reward_function = None
    transition_function = {}
    initial_states = None
    goal_states = []

    with open(problem_file, "r") as file:
        while True:
            line = file.readline()
            if not line: break # end of file

            if line == "":
                continue #empty line
            if line.startswith("states"):
                states = read_state_section(file)
            elif line.startswith("action"):
                action_name, transition_probabilities = read_action_section(file, line)
                transition_function[action_name] = transition_probabilities
            elif line.startswith("reward"):
                reward_function = read_reward_section(file)
            elif line.startswith("initialstate"):
                initial_states = read_initial_state_section(file)
            elif line.startswith("goalstate"):
                goal_states = read_goal_state_section(file)

    return EnumerativeMDP(states, reward_function, transition_function, initial_states, goal_states)