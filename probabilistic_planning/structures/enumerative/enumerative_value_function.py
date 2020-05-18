import numpy as np

class EnumerativeValueFunction:
    def __init__(self, states, values = None):
        heuristic_function = None
        internal_values_setted = False

        if values is None:
            heuristic_function = lambda state : 0
        elif callable(values):
            heuristic_function = values
        elif type(values) == EnumerativeValueFunction:
            self._indexed_states = values._indexed_states
            self._internal_matrix = values._internal_matrix.copy()
            internal_values_setted = True
        else:
            raise ValueError("values should be a function or a EnumerativeValueFunction")

        if not internal_values_setted:
            self._internal_matrix = np.zeros(( len(states), 1 ))
            self._indexed_states = {}
            for index, state in enumerate(states):
                self._indexed_states[state] = index
                self._internal_matrix[index] = heuristic_function(state)

    def __getitem__(self, state):
        state_index = self._indexed_states[state]
        return self._internal_matrix[state_index, 0]

    def __setitem__(self, state, value):
        state_index = self._indexed_states[state]
        self._internal_matrix[state_index, 0] = value

    def __repr__(self):
        dict_representation = {}

        for state in self._indexed_states.keys():
            state_index = self._indexed_states[state]
            dict_representation[state] = self._internal_matrix[state_index]

        return repr(dict_representation)

    def copy(self):
        states = list(self._indexed_states.keys())
        return EnumerativeValueFunction(states, self)

    def to_matrix(self):
        return self._internal_matrix
