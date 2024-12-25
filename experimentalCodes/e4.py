
import torch 
from odk_backend import replace_ops_with_trtion



def generate_data(b):
    return (
        torch.randn(b, 3, 128, 128).to(torch.float32).cuda(),
        torch.randint(1000, (b,)).cuda(),
    )

class my_mod(torch.nn.Module):
    def __init__(self,) -> None:   
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, 3)
        self.conv2 = torch.nn.Conv2d(3, 10, 3)
        
    def forward(self, x):
        o1 = self.conv1(x)
        o2 = self.conv2(x)
        o = o1 + o2
        return o
    
    
model = my_mod().cuda()
inp = generate_data(16)[0]
out = model(inp)
# Reset since we are using a different backend.
torch._dynamo.reset()

opt_model = torch.compile(model, backend=replace_ops_with_trtion)
out_opt = opt_model(inp)


print(torch.isclose(out_opt, out).all())