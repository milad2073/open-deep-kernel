
import torch 
import triton
import triton.language as tl
from torch.fx import passes, symbolic_trace
import operator

from typing import List



def replace_ops_with_trtion(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("custom backend called with FX graph:")
    gm.graph.print_tabular()
    # g = passes.graph_drawer.FxGraphDrawer(gm, 'resnet18')
    # with open("resnet18_compile.svg", "wb") as f:
    #     f.write(g.get_dot_graph().create_svg())
    return gm.forward



# Triton kernel for addition
@triton.jit
def triton_add_kernel(
    x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    result = x + y + 1
    tl.store(output_ptr + offsets, result, mask=mask)

def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.shape == y.shape, "Shapes of tensors must match"
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    triton_add_kernel[grid](
        x,
        y,
        output,
        n_elements,
        BLOCK_SIZE=1024
    )
    return output

# Custom backend for torch.compile
def replace_ops_with_trtion(graph_module: torch.fx.GraphModule, example_inputs: tuple):
    """A backend that replaces operations with their Triton-based equevalents."""
    
    graph = graph_module.graph
    print(set([ node.target.__name__ for node in graph.nodes if node.op=='call_function']))
    for node in graph.nodes:
        if node.op == 'call_function' and (node.target == operator.iadd or 
                                           node.target == operator.add  or
                                           node.target == torch.add       ):
            # Replace torch.add with Triton add implementation
            with graph.inserting_after(node):
                new_node = graph.create_node(
                    'call_function', triton_add, args=node.args, kwargs=node.kwargs
                )
                node.replace_all_uses_with(new_node)
            graph.erase_node(node)

    graph.lint()
    graph_module.recompile()
    return graph_module
