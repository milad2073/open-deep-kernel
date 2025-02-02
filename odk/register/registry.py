from .. import graph_passes

def add_latest_passes_to_docstribg(func):
    func.__doc__ = func.__doc__.replace('graph_passes.__all__',
                                        str(graph_passes.__all__))
    return func    

class Registry(dict):

    
    """A registry for storing and retrieving operations."""
    @add_latest_passes_to_docstribg
    def set(self, key: str):
        """
        Decorator to register a function as an operation in the Registry.

        This method takes a string key representing the operation name, 
        checks if it is a valid operator, and returns a decorator that 
        wraps a function. The wrapped function is then stored in the 
        Registry under the specified key after.

        Parameters:
        -----------
        key : str
            The name of the operation to register. This should match one 
            of the keys defined in `graph_passes.__all__`.

        Returns:
        --------
        function
            A decorator that wraps the function to be registered.

        Raises:
        -------
        KeyError
            If the provided key is not a valid operator name.

        Example:
        ---------
        @registry.set('relu')
        def my_relu_function(x):
            return max(0, x)
        """
        lower_key = key.lower()
        if lower_key not in graph_passes.__all__:
            raise KeyError("{} is an invalid operator name. The currently supported operators are: {}".format(lower_key, graph_passes.__all__))
        
        def wrap(func):
            pass_class = getattr(graph_passes, lower_key)
            self.__setitem__(lower_key, pass_class(func))
            return func
        
        return wrap
    