class EnumerativeValueFunction:
    def __init__(self, values = None):
        heuristic_function = None
        internal_dict = {}

        if values is None:
            heuristic_function = lambda state : 0
        elif callable(values):
            heuristic_function = values
        elif type(values) == EnumerativeValueFunction:
            heuristic_function = values._heuristic_function
            internal_dict = dict(values._internal_dict)
        else:
            raise ValueError("values should be a function or a EnumerativeValueFunction")

        self._heuristic_function = heuristic_function
        self._internal_dict = internal_dict

    def __getitem__(self, state):
        if state not in self._internal_dict.keys():
            self._internal_dict[state] = self._heuristic_function(state)

        return self._internal_dict[state]

    def __setitem__(self, state, value):
        self._internal_dict[state] = value

    def states(self):
        return list(self._internal_dict.keys())

    def copy(self):
        return EnumerativeValueFunction(self)
