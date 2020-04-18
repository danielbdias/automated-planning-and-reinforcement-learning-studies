def validate_defined_argument(argument_value, argument_name):
    """Validates if a given argument has a defined value (is not None)."""
    if argument_value is None:
        raise ValueError(f"The {argument_name} should be defined")
