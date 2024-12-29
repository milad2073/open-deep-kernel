from odk import Kernels, odk_backend
import triton
import triton.language as tl
import torch 
import torchvision.models as models 


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

# Setting the kernels
@Kernels.set_addaion
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

# Derfining the model
model = models.resnet18().cuda()

# replacing pytorch built-in kernels with defined kernels 
torch._dynamo.reset()
model_with_relaced_kernels = torch.compile(model, backend=odk_backend)




# generating data
def generate_data(b):
    return (
        torch.randn(b, 3, 299, 299).to(torch.float32).cuda(),
        torch.randint(1000, (b,)).cuda(),
    )

inp = generate_data(1)[0]


out_1 = model(inp)
out_2 = model_with_relaced_kernels(inp)

print(torch.isclose(out_1, out_2).all())