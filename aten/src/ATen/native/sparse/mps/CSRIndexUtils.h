#pragma once

#include <ATen/ATen.h>

namespace at::native::mps::csr {

// Builds prefix pointers (batch_ptr) for batched COO data where the first
// dimension of indices encodes the batch id. The returned tensor has shape
// [batch_count + 1] and dtype long on the same MPS device as the input.
Tensor build_batch_ptr_mps(const Tensor& batch_indices, int64_t batch_count);

// Writes prefix pointers (batch_ptr) for batched COO data where the first
// dimension of indices encodes the batch id. The provided output tensor must be
// allocated on the same MPS device with shape [batch_count + 1] and dtype long.
void build_batch_ptr_mps_out(
    const Tensor& batch_indices,
    int64_t batch_count,
    const Tensor& batch_ptr);

// Builds CSR-style row pointers per batch. The returned tensor has shape
// [batch_count * (rows_per_batch + 1)] and dtype long on the same MPS device as
// the inputs.
Tensor build_row_ptr_per_batch_mps(
    const Tensor& rows,
    const Tensor& batch_ptr,
    int64_t batch_count,
    int64_t rows_per_batch);

// Writes CSR-style row pointers per batch into the provided output tensor which
// must have shape [batch_count * (rows_per_batch + 1)] and dtype long on the
// same MPS device.
void build_row_ptr_per_batch_mps_out(
    const Tensor& rows,
    const Tensor& batch_ptr,
    int64_t batch_count,
    int64_t rows_per_batch,
    const Tensor& row_ptr);

// Expands batched CSR (crow_indices/col_indices) back to batched COO indices
// when the compressed dimension corresponds to rows. Returns a tensor of shape
// [batch_count, 2, nnz_per_batch] containing row/col pairs. The inputs must be
// contiguous MPS tensors with matching batch shapes, `rows_per_batch + 1`
// entries in crow_indices for each batch, and nnz_per_batch entries in
// col_indices. The resulting COO indices can be reshaped or flattened by the
// caller as needed.
Tensor expand_csr_rows_to_coo(
    const Tensor& crow_indices,
    const Tensor& col_indices,
    int64_t rows_per_batch,
    bool out_int32,
    bool transpose);

void expand_csr_rows_to_coo_out(
    const Tensor& crow_indices,
    const Tensor& col_indices,
    int64_t rows_per_batch,
    bool out_int32,
    bool transpose,
    const Tensor& coo_indices);

} // namespace at::native::mps::csr


