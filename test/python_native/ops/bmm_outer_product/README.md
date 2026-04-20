# BMM Outer Product Triton Implementation Tests

This directory contains tests for the Triton-optimized batch matrix multiplication (BMM) implementation specifically optimized for outer product cases.

## Operation Description

The Triton BMM outer product optimization targets a specific pattern in batch matrix multiplication:

- **Input A**: Shape `(B, M, 1)` - batch of column vectors
- **Input B**: Shape `(B, 1, N)` - batch of row vectors
- **Output**: Shape `(B, M, N)` - batch of outer products

This is mathematically equivalent to `torch.bmm(a, b)` but optimized for the outer product pattern where one matrix has width 1 and the other has height 1.

## Files

- `test_triton_bmm_outer_product.py`: Main test file containing BinaryUfuncInfo definition
- `__init__.py`: Package initialization
- `README.md`: This file

## Features Tested

### BinaryUfuncInfo DSL Integration
- Uses `BinaryUfuncInfo` class with `dsl_name='triton'` (the actual DSL backend name)
- Test variant name is `'triton_bmm_outer_product'` (specific test implementation)
- **Fully integrated with existing CI infrastructure** - automatically tested by standard `@ops(binary_ufuncs)` decorators
- Imported by `torch/testing/_internal/common_methods_invocations.py`

### Triton Optimization Conditions
The tests focus on conditions where Triton optimization is expected:

1. **3D tensors**: Both inputs must be 3-dimensional
2. **Outer product pattern**: A has shape `(B, M, 1)`, B has shape `(B, 1, N)`
3. **CUDA device**: Required for Triton execution
4. **Contiguous memory**: Required for efficient memory access
5. **Non-complex dtypes**: Real-valued tensors only
6. **Non-empty tensors**: Both tensors must have elements

### Test Coverage

#### Sample Inputs
- Various batch sizes (1, 2, 4, 8, 16 batches)
- Different matrix dimensions (16x32, 64x64, 128x256, 512x128, etc.)
- Non-power-of-2 dimensions for robustness (7x11, etc.)
- Various value ranges including negative values and small positive values
- Edge cases for numerical stability

#### Correctness Verification
- Reference implementation using `torch.bmm(a, b)`
- Tolerance settings appropriate for different floating point precisions
- Comparison against standard PyTorch BMM implementation
- Direct kernel testing bypasses dispatch system

#### Device and Precision Handling
- CUDA-only testing (Triton requirement)
- Appropriate tolerance overrides for different precisions (float16, bfloat16, float32, float64)
- Skips for unsupported configurations (complex dtypes, non-CUDA devices)

### Utility Functions

- `sample_inputs_triton_bmm_outer_product()`: Generates test inputs for outer product patterns
- `reference_bmm_outer_product()`: Reference implementation for numerical comparison
- `verify_outer_product_conditions()`: Helper to check if tensors meet optimization conditions
- `create_outer_product_tensors()`: Creates tensors meeting optimization conditions

## Integration

The OpInfo is automatically discovered and loaded via the auto-discovery system:

```python
# In torch/testing/_internal/common_methods_invocations.py
from python_native.ops import dsl_opinfos
op_db.extend(dsl_opinfos)
```

### Auto-Discovery System
DSL OpInfos are automatically discovered from the `test/python_native/ops/` directory structure:

```
test/python_native/ops/
├── __init__.py           # Auto-discovery logic
├── bmm_outer_product/
│   ├── __init__.py
│   ├── test_triton_bmm_outer_product.py  # Contains triton_bmm_outer_product_opinfo
│   └── README.md
├── silu/
│   └── test_triton_silu.py
└── [future_ops]/
    └── test_*.py        # Any test files with *_opinfo exports
```

### CI Integration
The DSL variant inherits from BinaryUfuncInfo and is automatically included in:
- `op_db` (699 total OpInfos)
- `binary_ufuncs` (BinaryUfuncInfo collection, including DSL variants)

This means **existing CI test classes with `@ops(binary_ufuncs)` decorators automatically test DSL variants** without requiring separate test infrastructure.

### Filtering Examples

```bash
# Test all OpInfos (698 + 2 DSL OpInfos)
python -c "from torch.testing._internal.common_methods_invocations import op_db; print(len(op_db))"

# Test only triton DSL OpInfos (2 OpInfos: BMM + SiLU)
OPINFO_RESTRICT_TO_DSL=triton python -c "from torch.testing._internal.common_methods_invocations import op_db; print(len(op_db))"

# Test only binary operations with DSL filtering
OPINFO_RESTRICT_TO_DSL=triton python -c "from torch.testing._internal.common_methods_invocations import binary_ufuncs; print([op.name for op in binary_ufuncs if hasattr(op, 'dsl_name')])"

# Invalid - triton_bmm_outer_product is not a DSL name (0 OpInfos)
OPINFO_RESTRICT_TO_DSL=triton_bmm_outer_product python -c "from torch.testing._internal.common_methods_invocations import op_db; print(len(op_db))"
```

## Usage

The tests run as part of the standard PyTorch OpInfo test suite when:
1. CUDA is available
2. The test directory is in Python path
3. PyTorch is built with Triton support

## Implementation Details

The BinaryUfuncInfo tests the implementation in:
- `torch/_native/ops/bmm_outer_product/triton_impl.py`: Dispatch logic and optimization conditions
- `torch/_native/ops/bmm_outer_product/triton_kernels.py`: Triton kernel implementations

### Architecture Summary

- **Base class**: `BinaryUfuncInfo` (inherits from `OpInfo`)
- **DSL identification**: `dsl_name='triton'` parameter
- **CI integration**: Automatically included in `binary_ufuncs` collection
- **Direct testing**: Tests `bmm_outer_product` function directly
- **Filtering**: `OPINFO_RESTRICT_TO_DSL=triton` works correctly

### Optimization Pattern

The Triton kernel recognizes and optimizes this specific BMM pattern:

```python
# Pattern: (B, M, 1) @ (B, 1, N) -> (B, M, N)
a = torch.randn(4, 32, 1, device='cuda')  # Column vectors
b = torch.randn(4, 1, 64, device='cuda')  # Row vectors
result = torch.bmm(a, b)                  # Shape: (4, 32, 64)

# This gets optimized by Triton as outer products:
# result[i] = a[i] @ b[i] = a[i, :, 0:1] @ b[i, 0:1, :]
```

The tests validate both the optimization conditions and numerical correctness of the Triton-accelerated BMM implementation while ensuring seamless integration with existing CI infrastructure.