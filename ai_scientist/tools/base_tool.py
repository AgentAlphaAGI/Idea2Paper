from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseTool(ABC):
    """
    An abstract base class for defining custom tools.

    Attributes:
    -----------
    - name (str): The name of the tool.
    - description (str): A short description of what the tool does.
    - parameters (list): A list of parameters that the tool requires, each parameter should be a dictionary with 'name', 'type', and 'description' key/value pairs.

    Usage:
    ------
    To use this class, you should subclass it and provide an implementation for the `use_tool` abstract method.
    """

    def __init__(self, name: str, description: str, parameters: List[Dict[str, Any]]):
        self.name = name
        self.description = description
        self.parameters = parameters

    @abstractmethod
    def use_tool(self, **kwargs) -> Any:
        """Abstract method that should be implemented by subclasses to define the functionality of the tool."""
        pass
