"""Structures to handle decision digrams, usually used to handle symbolic representations (like factored MDP).

ADD: represents an algebraic decision diagram, commonly used to represent factored functions
"""

# for now, I'm masking the use of pyddlib for tests purposes

from pyddlib.add import ADD

__all__ = ["ADD"]
