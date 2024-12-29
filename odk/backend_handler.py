import torch 
from .register import Kernels
from .graph_transformations.base import baseTransformation


# It is a custom backend for torch.compile   
def odk_backend(graph_module: torch.fx.GraphModule, example_inputs: tuple):
    """A backend that replaces operations with their Triton-based equevalents."""
    
    graph = graph_module.graph
    
    for node in graph.nodes:
        kernel:baseTransformation
        for kernel_name, kernel in Kernels.items():
            if (node.target in kernel.operators):
                kernel.replacement(graph, node)
                
    graph.lint()
    graph_module.recompile()
    return graph_module
 