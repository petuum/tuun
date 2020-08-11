"""
Abstract classes for backends.
"""

from abc import ABC, abstractmethod


class Backend(ABC):
    """Abstract class for backend tuning system."""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def minimize_function(self):
        pass

    @abstractmethod
    def suggest_to_minimize(self):
        pass
