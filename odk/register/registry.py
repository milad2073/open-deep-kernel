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
        self.__setitem__('addation', addation(func))
        return func
    
    
    