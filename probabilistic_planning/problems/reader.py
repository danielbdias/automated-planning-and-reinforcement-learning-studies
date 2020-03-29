from probabilistic_planning.structures.mdp import EnumerativeMDP

def clean_tabs(value):
    return value.replace("\t", "")


def clean_spaces(value):
    return value.replace(" ", "")


def clean_line_breaks(value):
    return value.replace("\r", "").replace("\n", "")


def read_state_section(file):
    line = file.readline()
    line = clean_spaces(line)
    line = clean_tabs(line)
    line = clean_line_breaks(line)

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
    action_name = clean_line_breaks(data[1])

    while True:
        line = file.readline()

        if not line:
            raise Exception("endaction token not found !")

        if line.startswith("endaction"):
            return action_name, transition_probabilities

        line = clean_tabs(line)
        data = line.split(" ")

        from_state = data[0]
        to_state = data[1]
        probability = float(data[2])

        if not transition_probabilities.has_key(from_state):
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

        line = clean_tabs(line)
        data = line.split(" ")

        state = data[0]
        value = int(data[1])

        reward[state] = value


def read_initial_state_section(file):
    initial_states = []

    while True:
        line = file.readline()

        if not line:
            raise Exception("endinitialstate token not found !")

        if line.startswith("endinitialstate"):
            return initial_states

        state = clean_tabs(line)
        initial_states.append(state)


def read_goal_state_section(file):
    goal_states = []

    while True:
        line = file.readline()

        if not line:
            raise Exception("endgoalstate token not found !")

        if line.startswith("endgoalstate"):
            return goal_states

        state = clean_tabs(line)
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