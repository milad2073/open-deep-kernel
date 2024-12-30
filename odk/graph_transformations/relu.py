from .base import baseTransformation
import operator
import torch



class relu(baseTransformation):
    
    
    
    def __init__(self, func):
        super().__init__()
        self.func = func
    
    def is_applicable(self, node):
        if node.op == "call_module":
            return node.meta['source_fn'][1] == torch.nn.modules.activation.ReLU
        return False
    
    def replacement (self, graph, node):
        with graph.inserting_after(node):
            new_node = graph.create_node(
            'call_function', self.func, args=node.args, kwargs=node.kwargs
            )
            node.replace_all_uses_with(new_node)
            graph.erase_node(node)
    
    
