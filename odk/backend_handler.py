import torch 
from .register.registry import Registry
from .graph_transformations.base import baseTransformation
from pathlib import Path
from torch.fx.passes.graph_drawer import FxGraphDrawer
from  datetime import datetime
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ODKBackend")
logger.setLevel(logging.DEBUG)

# It is a custom backend for torch.compile   
class ODKBackend:
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
            svg_filename = datetime.now().strftime('%Y%m%d_%H%M%S%f')
            saved_file = self._draw_graphes(graph_module, f"{svg_filename}_prev")
            if saved_file:
                logger.info(f'Graph of the inital module saved in {saved_file}')
            
        graph = graph_module.graph
        for node in graph.nodes:
            transformation:baseTransformation
            for transformation_name, transformation in self.kernel_dict.items():
                if transformation.is_applicable(node):
                    transformation.replacement(graph, node)
                    logger.debug(f"Node Replacement: transformation type -> '{transformation_name}'  "+
                                 f"node name -> '{node.name}'")
                    
        graph.lint()
        graph_module.recompile()
        
        if self.draw_graphes:
            saved_file = self._draw_graphes(graph_module, f"{svg_filename}_post")
            if saved_file:
                logger.info(f'Graph of the final module saved in {saved_file}')
        
        return graph_module
    
    def _draw_graphes(self, graph_module, name):
        path = self.result_path /  f"{name}.svg"
        try:
            with open(path, "wb") as f:
                svg_file = FxGraphDrawer(graph_module,'res').get_dot_graph().create_svg()
                f.write(svg_file)
            return path
        except Exception as e:
            logger.warning(f'Cannot save the file in {path}')
            logger.warning(e)
            return None
        
        
    