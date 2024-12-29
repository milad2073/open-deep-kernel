from abc import abstractmethod

class baseTransformation:
    
    
    @property
    @abstractmethod
    def operators(self): 
        pass
    
    
    @abstractmethod
    def replacement(self, graph, node):
        pass