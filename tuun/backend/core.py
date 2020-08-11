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
    def tune_function(self):
        pass

    @abstractmethod
    def suggest(self):
        pass
