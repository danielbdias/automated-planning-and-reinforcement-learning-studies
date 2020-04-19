"""Structures to represent different types of MDP.

EnumerativeMDP: represents a MDP with enumerable states
FactoredMDP: represents a MDP factored by state variables
"""

from .enumerative.enumerative_mdp import EnumerativeMDP
from .factored.factored_mdp import FactoredMDP

__all__ = ["EnumerativeMDP", "FactoredMDP"]
