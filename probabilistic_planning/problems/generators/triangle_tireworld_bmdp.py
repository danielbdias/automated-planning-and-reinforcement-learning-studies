from argparse import ArgumentParser
import os, sys

mark_car_location = "car-at-x%02dy%02d"
mark_flattired = "-flattired"
mark_hasspares = "-hasspare"

class State(object):
    def __init__(self, x, y, flattired, hasspare):
        self.x = x
        self.y = y
        self.flattired = flattired
        self.hasspare = hasspare
        
def configure_args():
    """Define and validate the program arguments."""
    
    parser = ArgumentParser(description='Generate an instance of the triangle-tireworld MDP domain used in IPPC 2009.')
    parser.add_argument("-instance_number", type=int, help = "Instance number, that define the grid size.")
    parser.add_argument("-bound", type=float, help = "Bound size.")
    parser.add_argument("-output", type=str, help = "Name of the domain output file.")

    args = parser.parse_args()

    if args.instance_number < 0:
        print "The 'instancenumber' parameter must be equal or greater than 1 (one)."
        sys.exit(-1)
    elif args.bound < 0.0 or args.bound > 1.0:
        print "The 'bound' parameter must have a value between 0.0 and 1.0"
        sys.exit(-1)
    elif os.path.isfile(args.output):
        print "The file described in the 'output' parameter will be overriden."        

    return args

def get_state_string(state):
    state_as_string = mark_car_location %(state.x, state.y)

    if state.flattired:
        state_as_string = state_as_string + mark_flattired

    if state.hasspare:
        state_as_string = state_as_string + mark_hasspares
    
    return state_as_string

def get_grid_size(instance_number):
    width = 3 + (instance_number - 1)
    height = 3 + (instance_number * 2)
    
    return width, height
  
def get_valid_grid_coordinates(instance_number):
    width, height = get_grid_size(instance_number)
    
    coordinates = []

    for x in xrange(1, width + 1):
        for y in xrange(x, (height + 1) - x + 1, 2):
            coordinates.append( ( x, y ) )
            
    return coordinates
  
def enumerate_states(instance_number):
    for x, y in get_valid_grid_coordinates(instance_number):
        yield State(x, y, False, False)
        yield State(x, y, True, False)
        yield State(x, y, False, True)
        yield State(x, y, True, True)

def get_states(instance_number):
    states = []

    for state in enumerate_states(instance_number):
        states.append(get_state_string(state))
    
    return states

def is_goal_state(instance_number, state):
    width, height = get_grid_size(instance_number)
    width = 1 #ignore the previous value 
    
    return (state.y == height and state.x == 1)
    
def get_flattire_probability(x, max_x, bound):
    first_point = 0.1
    last_point = 0.9
    
    lower_bound = first_point + (x - 1) * ((last_point - first_point) / (max_x - 1))
    upper_bound = lower_bound + bound
    
    if upper_bound > 1.0:
        upper_bound = 1.0
        
    return lower_bound, upper_bound

def get_movenorth_action(instance_number, bound):
    action = []

    width, height = get_grid_size(instance_number)
    valid_coordinates = get_valid_grid_coordinates(instance_number)
    
    for state in enumerate_states(instance_number):
        state_as_string = get_state_string(state)  
    
        has_north_cell = ( state.x, state.y + 2 ) in valid_coordinates
    
        if state.flattired or is_goal_state(instance_number, state) or not has_north_cell: #remains in same state
            action.append( (state_as_string, state_as_string, 1.0, 1.0) )
        else:
            lower_probability, upper_probability = get_flattire_probability(state.y + 2, height, bound)           
            
            target_state_as_string = get_state_string(State(state.x, state.y + 2, False, state.hasspare))           
            action.append( (state_as_string, target_state_as_string, 1.0 - upper_probability, 1.0 - lower_probability) )
            
            target_state_as_string = get_state_string(State(state.x, state.y + 2, True, state.hasspare))           
            action.append( (state_as_string, target_state_as_string, lower_probability, upper_probability) )

    return action

def get_movenortheast_action(instance_number, bound):
    action = []

    width, height = get_grid_size(instance_number)
    valid_coordinates = get_valid_grid_coordinates(instance_number)
    
    for state in enumerate_states(instance_number):
        state_as_string = get_state_string(state)  
    
        has_northeast_cell = ( state.x + 1, state.y + 1 ) in valid_coordinates
    
        if state.flattired or is_goal_state(instance_number, state) or not has_northeast_cell: #remains in same state
            action.append( (state_as_string, state_as_string, 1.0, 1.0) )
        else:
            lower_probability, upper_probability = get_flattire_probability(state.y + 1, height, bound)           
            
            target_state_as_string = get_state_string(State(state.x + 1, state.y + 1, False, state.hasspare))           
            action.append( (state_as_string, target_state_as_string, 1.0 - upper_probability, 1.0 - lower_probability) )
            
            target_state_as_string = get_state_string(State(state.x + 1, state.y + 1, True, state.hasspare))           
            action.append( (state_as_string, target_state_as_string, lower_probability, upper_probability) )

    return action

def get_movenorthwest_action(instance_number, bound):
    action = []

    width, height = get_grid_size(instance_number)
    valid_coordinates = get_valid_grid_coordinates(instance_number)
    
    for state in enumerate_states(instance_number):
        state_as_string = get_state_string(state)  
    
        has_northwest_cell = ( state.x - 1, state.y + 1 ) in valid_coordinates
    
        if state.flattired or is_goal_state(instance_number, state) or not has_northwest_cell: #remains in same state
            action.append( (state_as_string, state_as_string, 1.0, 1.0) )
        else:
            lower_probability, upper_probability = get_flattire_probability(state.y + 1, height, bound)           
            
            target_state_as_string = get_state_string(State(state.x - 1, state.y + 1, False, state.hasspare))           
            action.append( (state_as_string, target_state_as_string, 1.0 - upper_probability, 1.0 - lower_probability) )
            
            target_state_as_string = get_state_string(State(state.x - 1, state.y + 1, True, state.hasspare))           
            action.append( (state_as_string, target_state_as_string, lower_probability, upper_probability) )

    return action

def get_changetire_action(instance_number):
    action = []
    
    for state in enumerate_states(instance_number):
        state_as_string = get_state_string(state)  
        
        if is_goal_state(instance_number, state) or not state.flattired or not state.hasspare: #remains in same state
            action.append( (state_as_string, state_as_string, 1.0, 1.0) )
        else:
            target_state_as_string = get_state_string(State(state.x, state.y, False, False))           
            action.append( (state_as_string, target_state_as_string, 1.0, 1.0) )
    
    return action

def get_loadtire_action(instance_number):
    action = []
    
    for state in enumerate_states(instance_number):
        state_as_string = get_state_string(state)  
        
        if is_goal_state(instance_number, state) or state.hasspare or state.x == 1: #remains in same state
            action.append( (state_as_string, state_as_string, 1.0, 1.0) )
        else:
            target_state_as_string = get_state_string(State(state.x, state.y, state.flattired, True))           
            action.append( (state_as_string, target_state_as_string, 1.0, 1.0) )
    
    return action
    
def get_actions(instance_number, bound):
    actions = {}

    actions["move-north"] = get_movenorth_action(instance_number, bound)
    actions["move-northeast"] = get_movenortheast_action(instance_number, bound)
    actions["move-northwest"] = get_movenorthwest_action(instance_number, bound)
    actions["change-tire"] = get_changetire_action(instance_number)
    actions["load-tire"] = get_loadtire_action(instance_number)
    
    return actions

def get_reward(instance_number):
    reward = []
    
    for state in enumerate_states(instance_number):
        state_as_string = get_state_string(state)
        
        if is_goal_state(instance_number, state):
            reward.append( (state_as_string, 0) )
        else:
            reward.append( (state_as_string, -1) )
                
    return reward

def get_cost():
    cost = []

    cost.append( ( "move-north", 0 ) )
    cost.append( ( "move-northeast", 0 ) )
    cost.append( ( "move-northwest", 0 ) )
    cost.append( ( "change-tire", 0 ) )
    cost.append( ( "load-tire", 0 ) )
    
    return cost

def get_discount_factor():
    return 0.9

def get_initial_states(instance_number):
    return [ get_state_string(State(1, 1, False, True)) ] 

def get_goal_states(instance_number):
    width, height = get_grid_size(instance_number)

    return [ 
        get_state_string(State(1, height, False, False)),
        get_state_string(State(1, height, False, True)),
        get_state_string(State(1, height, True, False)),
        get_state_string(State(1, height, True, True))
    ] 
    
def generate_domain_data(instance_number, bound):
    states = get_states(instance_number)

    actions = get_actions(instance_number, bound)

    reward = get_reward(instance_number)

    cost = get_cost()

    discount_factor = get_discount_factor()

    initial_states = get_initial_states(instance_number)
    
    goal_states = get_goal_states(instance_number)
    
    domain_data = []

    domain_data.append("states")  
    domain_data.append("\t" + ", ".join(states))
    domain_data.append("endstates")
    domain_data.append("")

    for action_name in actions.keys():
        domain_data.append("action %s" %action_name)
        for from_state, to_state, lower_probability, upper_probability in actions[action_name]:
            domain_data.append("\t%s %s %f %f" %(from_state, to_state, lower_probability, upper_probability))
        domain_data.append("endaction")
        domain_data.append("")

    domain_data.append("reward")
    for state, value in reward:
        domain_data.append("\t%s %d" %(state, value))
    domain_data.append("endreward")
    domain_data.append("")

    domain_data.append("cost")
    for action, value in cost:
        domain_data.append("\t%s %d" %(action, value))
    domain_data.append("endcost")
    domain_data.append("")

    domain_data.append("discount factor %f" %discount_factor)

    domain_data.append("initialstate")
    for state in initial_states:
        domain_data.append("\t%s" %state)
    domain_data.append("endinitialstate")
    domain_data.append("")
    
    domain_data.append("goalstate")
    for state in goal_states:
        domain_data.append("\t%s" %state)
    domain_data.append("endgoalstate")
    
    return domain_data

def write_domain_file(domain_data, output):
    """Given a domain data structured in lines, write a file in output path with these lines."""
    
    with open(output, "wb") as f:
        for line in domain_data:
            f.write(line)
            f.write("\r\n")

#Start the program execution
if __name__ == "__main__":
    args = configure_args()

    domain_data = generate_domain_data(args.instance_number, args.bound)

    write_domain_file(domain_data, args.output)
