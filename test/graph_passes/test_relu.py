from odk import Registry, ODKBackend
import triton
import triton.language as tl
import torch 
import torchvision.models as models 
import pytest 



############## DEFINE KERNELS  ##############

@triton.jit
def relu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.maximum(x, 0.0)
    tl.store(y_ptr + offsets, y, mask=mask)
             
Kernels = Registry()


@Kernels.set('relu')
def triton_relu(x):
    y = torch.empty_like(x)
    N = x.numel()
    BSIZE = 1024
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, y, N, BLOCK_SIZE=BSIZE)
    return y




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

