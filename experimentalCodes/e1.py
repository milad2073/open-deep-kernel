import torch
import triton
import triton.language as tl

# Triton kernel for addition
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

# The single class implementation
class TritonAddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, *, alpha=1, out=None):
        # Handle scalars by promoting to tensors
        if isinstance(x, (int, float)):
            x = torch.tensor(x, dtype=y.dtype, device=y.device)
        if isinstance(y, (int, float)):
            y = torch.tensor(y, dtype=x.dtype, device=x.device)

        # Ensure dtypes are compatible
        common_dtype = torch.promote_types(x.dtype, y.dtype)
        x = x.to(common_dtype)
        y = y.to(common_dtype)

        # Ensure devices match
        if x.device != y.device:
            raise RuntimeError("Input tensors must be on the same device")

        # Handle broadcasting
        x, y = torch.broadcast_tensors(x, y)

        # Ensure tensors are contiguous
        if not x.is_contiguous():
            x = x.contiguous()
        if not y.is_contiguous():
            y = y.contiguous()

        # Handle the alpha parameter
        y = y * alpha

        # Allocate output tensor
        output = torch.empty_like(x) if out is None else out

        # Launch Triton kernel
        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

        # Save tensors for backward pass
        ctx.save_for_backward(x, y)
        ctx.alpha = alpha
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Gradients for both x and y are the incoming gradient
        x, y = ctx.saved_tensors
        grad_x = grad_output
        grad_y = grad_output * ctx.alpha
        return grad_x, grad_y, None, None

# Wrapper class for convenience
class TritonAdd:
    @staticmethod
    def apply(x, y, *, alpha=1, out=None):
        return TritonAddFunction.apply(x, y)

    @staticmethod
    def monkey_patch():
        # Backup original torch.add
        TritonAdd._original_add = torch.add

        # Override torch.add
        def triton_add(x, y, *args, **kwargs):
            return TritonAdd.apply(x, y, **kwargs)

        torch.add = triton_add

        # Override the "+" operator
        # torch.Tensor.__add__ = lambda self, other: torch.add(self, other)

    @staticmethod
    def restore():
        # Restore original torch.add and "+"
        torch.add = TritonAdd._original_add
        # torch.Tensor.__add__ = lambda self, other: TritonAdd._original_add(self, other)


# Example Usage
x = torch.randn(1024, device='cuda')
y = torch.randn(1024, device='cuda')

# Patch PyTorch to use TritonAdd globally
TritonAdd.monkey_patch()

# Test addition
result_plus = x + y
result_add = torch.add(x, y, alpha=2)

# Validate
print(torch.allclose(result_plus, result_add))

# Restore original behavior if needed
TritonAdd.restore()
