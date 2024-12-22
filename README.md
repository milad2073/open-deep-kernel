# OpenDeepKernel

**OpenDeepKernel** is a flexible, high-performance deep learning framework built to enable practitioners and researchers to experiment with custom kernel implementations and optimizations. Leveraging **Triton-lang** for JIT-compiled GPU kernels, OpenDeepKernel provides full control over the kernel code, allowing users to easily integrate, test, and optimize custom kernels for both training and inference. This makes it ideal for users who want to explore the deep learning process at the kernel level and leverage GPU acceleration for optimized performance.

## Features

- **Customizable Kernels:** Implement your own deep learning kernels using Triton-lang and replace the default ones.
- **Kernel-Level Experimentation:** Flexibly experiment with different kernels to optimize performance for a wide range of tasks.
- **GPU Acceleration:** Built for high-performance, GPU-accelerated deep learning workflows using **Triton**.
- **Simple API:** Provides a simple and extensible API to customize and integrate kernels into your models.
- **Optimized Training and Inference:** Efficient execution of deep learning models with support for various GPU models.

## Installation

To install **OpenDeepKernel**, you can clone the repository and install the dependencies using `pip`:

```bash
git clone https://github.com/milad2073/open-deep-kernel.git
cd open-deep-kernel
pip install .
```

## Getting Started
Define Custom Kernels
In OpenDeepKernel, users can define custom kernels by writing their own kernel code and integrating it into the training or inference pipeline. To define a custom kernel, follow these steps:

Create a new kernel file in the kernels directory.
Implement the kernel logic in the file.
Integrate the kernel by specifying it in the model configuration or during training.
Example:
```python
import opendeepkernel as odk

# Define a custom kernel function
def custom_kernel(input_tensor):
    # Implement custom logic
    return output_tensor

# Use the custom kernel in a model
model = odk.Model(kernel=custom_kernel)
model.train()
```

## Experiment with Different Kernels
You can experiment with different kernels in a modular fashion. The framework allows you to swap kernels at various stages of the model's pipeline (e.g., during forward pass, backpropagation, etc.).
```python
model.set_kernel('custom_kernel')
```

## Training and Inference
Once your custom kernel is defined, you can start training your models using the built-in training loop:
```python
model.train()
```
For inference:
```python
predictions = model.infer(input_data)
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




