import torch 
import triton
import torchvision.models as models 
from torch.fx import passes, symbolic_trace

model = models.resnet18()

gm = symbolic_trace(model)


g = passes.graph_drawer.FxGraphDrawer(gm, 'resnet18')
with open("resnet18.svg", "wb") as f:
    f.write(g.get_dot_graph().create_svg())