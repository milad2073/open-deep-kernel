import torch
from ..graph_transformations.addation import addation


class Registry(dict):
    
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Registry, cls).__new__(cls)
        return cls.instance
    
    def __init__(self) -> None:
        self.kernels = {}
        
    
    
    def set_addaion(self,func): 
        """
        Decorator to register a function as an 'addation' operation in the Registry.

        This method takes a function as an argument, wraps it with the `addation` 
        transformation, and stores it in the Registry under the key 'addation'. 
        This allows for easy retrieval and application of the 'addation' operation 
        later in the program.

        --------
        @registry.set_addaion
        
        def my_addition_function(x, y):
            pass
        """
        self.__setitem__('addation', addation(func))
        return func
    
    
    