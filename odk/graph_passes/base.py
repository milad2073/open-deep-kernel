from abc import abstractmethod

class basePass:
    """
    Abstract base class for transformations applied to graphs.
    This class defines the interface for all graph transformation classes.
    """

    @abstractmethod
    def replacement(self, graph, node):
        """
        Defines how a specific node in the graph should be replaced.

        Parameters:
        graph: The graph in which the node exists.
        node: The node to be replaced.

        Returns:
        bool: Whether the replacement was successful or not
        """
        pass   

    @abstractmethod
    def is_applicable(self, node):
        """
        Checks if the transformation can be applied to a given node.

        Parameters:
        node: The node to check.

        Returns:
        bool: True if the transformation can be applied, False otherwise.
        """
        pass
