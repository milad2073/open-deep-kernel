# OpenDeepKernel

**OpenDeepKernel** is a flexible framework built to enable practitioners and researchers to experiment with custom triton kernel implementations and optimizations. Leveraging **Triton-lang** for JIT-compiled GPU kernels, OpenDeepKernel provides full control over the kernel code, allowing users to easily integrate, test, and optimize custom kernels for both training and inference. This makes it ideal for users who want to explore the deep learning process at the kernel level and leverage GPU acceleration for optimized performance.

## Features

- **Customizable Kernels:** Implement your own deep learning kernels using Triton-lang and replace the default ones.
- **Kernel-Level Experimentation:** Flexibly experiment with different kernels to optimize performance for a wide range of tasks.
- **Simple API:** Provides a simple and extensible API to customize and integrate kernels into your models.


## Installation

To install **OpenDeepKernel**, you can clone the repository and install the dependencies using `pip`:

```bash
git clone https://github.com/milad2073/open-deep-kernel.git
cd open-deep-kernel
pip install .
```


## Experiment with Different Kernels
You can experiment with different kernels in a modular fashion.
#### Define custom kernel(s)
Implement your custom kernels using Triton-lang. Here's a simple example of a custom kernel:


```python
import triton
import triton.language as tl

@triton.jit
def relu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.maximum(x, 0.0)
    tl.store(y_ptr + offsets, y, mask=mask)
```


#### Register your custom kernel(s) 
Create a registry objects that collects your custom kernels

```python
from odk import Registry
import torch 

Kernels = Registry()

# 1) making a Registy object
Kernels = Registry()

# 2) adding the kernels to the registry
@Kernels.set_relu
def triton_relu(x):
    y = torch.empty_like(x)
    N = x.numel()
    BSIZE = 1024
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, y, N, BLOCK_SIZE=BSIZE)
    return y

```

#### Replace pytorch default operators
Using ODKBackend class, create a custom "torch.compile" backend that can replace pytorch default operations with your triton kernels.  
```python
from odk import ODKBackend

## create a backend 
my_backecnd = ODKBackend(Kernels, draw_graphes=True)
# replacing pytorch built-in kernels with defined kernels 
torch._dynamo.reset()
model_with_relaced_kernels = torch.compile(model, backend=my_backecnd)
```

Now use the created backend to modify any pytorch models. 

```python
import torchvision.models as models 

## Defining the model
model = models.resnet18().cuda()

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
```


## Documentation
The full documentation for OpenDeepKernel will be available soon. For now, please refer to the code and examples to get started.

## Contributing
We welcome contributions to OpenDeepKernel! If you'd like to improve the framework or add new features, feel free to fork the repository and submit a pull request.

Before contributing, please read our Code of Conduct and Contributing Guidelines.

## License
OpenDeepKernel is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
The design and architecture of this framework are inspired by the need for flexibility in deep learning kernel development.
Special thanks to the open-source community for supporting GPU acceleration and deep learning tools.




