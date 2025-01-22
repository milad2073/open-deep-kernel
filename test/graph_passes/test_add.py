from odk import Registry, ODKBackend
import triton
import triton.language as tl
import torch 
import torchvision.models as models 
import pytest 



############## DEFINE KERNELS  ##############

@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)
             
Kernels = Registry()

def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.shape == y.shape, "Shapes of tensors must match"
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](
        x,
        y,
        output,
        n_elements,
        BLOCK_SIZE=1024
    )
    return output





############## DEFINE PARAMETER LISTS  ##############

SHAPES = [[2,5],
          [2000,5000],
          [2,5,5,7,1,1,2]]

TYPES = [torch.float32, torch.float64, torch.float16]



@pytest.mark.parametrize( "type",  TYPES) 
@pytest.mark.parametrize( "shape",  SHAPES) 
def test_result(shape, type):
    def model(x,y):
        return x + y 

    my_backecnd = ODKBackend(Kernels, draw_graphes=True)

    torch._dynamo.reset()
    model_with_replaced_kernels = torch.compile(model, backend=my_backecnd)
    
    inp1 = torch.randn(shape,dtype=type).cuda()
    inp2 = torch.randn(shape,dtype=type).cuda()
    
    out_1 = model(inp1,inp2)
    out_2 = model_with_replaced_kernels(inp1,inp2)

    assert out_1.dtype == out_2.dtype , "dtypes are different"
    
    assert out_1.shape == out_2.shape , "shapes are different"
    
    assert torch.isclose(out_1, out_2).all() , "results are defferent"

