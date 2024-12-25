
import torch 
import torchvision.models as models 
from odk_backend import replace_ops_with_trtion

torch._C._jit_set_profiling_mode(True)
torch._C._jit_set_profiling_executor(True)


def generate_data(b):
    return (
        torch.randn(b, 3, 299, 299).to(torch.float32).cuda(),
        torch.randint(1000, (b,)).cuda(),
    )

model = models.resnet18().cuda()
inp = generate_data(1)[0]
out = model(inp)


traced_model = torch.jit.trace(model, inp)


# Reset since we are using a different backend.
torch._dynamo.reset()

opt_model = torch.compile(model, backend=replace_ops_with_trtion)
out_opt = opt_model(inp)


print(torch.isclose(out_opt, out).all())