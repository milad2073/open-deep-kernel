from odk.register import Kernels
from odk import odk_backend
import triton
import triton.language as tl
import torch 
import torchvision.models as models 

# Setting the kernels
@Kernels.set_addaion
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