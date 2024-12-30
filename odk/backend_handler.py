import torch 
from .register.registry import Registry
from .graph_transformations.base import baseTransformation
from pathlib import Path
from torch.fx.passes.graph_drawer import FxGraphDrawer


# It is a custom backend for torch.compile   
class Backend:
    def __init__(self, kernel_dict: Registry, draw_graphes=False, result_path=None) -> None:
        self.draw_graphes = draw_graphes
        self.kernel_dict = kernel_dict
        if result_path==None:
            self.result_path = Path.home() / ".odk"
        else:
            self.result_path = result_path
        
        Path(self.result_path).mkdir(parents=True, exist_ok=True)
        
    
    def __call__(self, graph_module: torch.fx.GraphModule, example_inputs: tuple):

        """A backend that replaces operations with their Triton-based equevalents."""
        
        if self.draw_graphes:
            self._draw_graphes(graph_module, "prev")
            
        graph = graph_module.graph
        for node in graph.nodes:
            kernel:baseTransformation
            for kernel_name, kernel in self.kernel_dict.items():
                if (node.target in kernel.operators):
                    kernel.replacement(graph, node)
                    
        graph.lint()
        graph_module.recompile()
        if self.draw_graphes:
            self._draw_graphes(graph_module, "post")
        return graph_module
    
    def _draw_graphes(self, graph_module, name):
        with open(self.result_path /  f"{name}.svg", "wb") as f:
            f.write(FxGraphDrawer(graph_module,'res').get_dot_graph().create_svg())
        
    