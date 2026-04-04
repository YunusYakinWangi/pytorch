# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

"""
Tests exposing places where _StridedShard is not handled alongside Shard.
Each test FAILs on unfixed code and PASSes after the fix.

Bug sites covered:
1. DTensorSpec.num_shards — .is_shard() misses _StridedShard
2. DTensorSpec.dim_map — .is_shard() misses _StridedShard
3. DTensorSpec.num_shards_map — .is_shard() misses _StridedShard
4. is_trivial_shard — isinstance(p, Shard) misses _StridedShard
5. _scaled_mm_scale_placement — isinstance(_, Shard) misses _StridedShard

Bug sites NOT covered (performance-only, no correctness impact):
- merge_placement in _tensor_ops.py — nested function, causes suboptimal
  strategy selection but redistribution masks it
- single_dim_strategy.py cost computation — inflated comm_bytes_gb estimate,
  doesn't change correctness
- _create_partial_input in strategy_validation.py — fallback at line 499
  handles _StridedShard correctly via distribute_tensor
"""

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard
from torch.distributed.tensor.placement_types import _StridedShard
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


class TestStridedShardDTensorSpec(DTensorTestBase):
    """Unit tests for DTensorSpec properties with _StridedShard placements."""

    @property
    def world_size(self):
        return 4

    @with_comms
    def test_num_shards_with_strided_shard(self):
        """DTensorSpec.num_shards must count _StridedShard placements.

        num_shards uses .is_shard() which returns False for _StridedShard,
        so num_shards returns 1 instead of the actual number of shards.
        """
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        full = torch.randn(4, self.world_size * 2, 6, device=self.device_type)
        dt = distribute_tensor(full, mesh, [Shard(1)])
        dt_flat = dt.flatten(0, 1)

        self.assertIsInstance(dt_flat.placements[0], _StridedShard)
        self.assertEqual(dt_flat._spec.num_shards, self.world_size)

    @with_comms
    def test_dim_map_with_strided_shard(self):
        """DTensorSpec.dim_map must map _StridedShard dims to mesh dims.

        dim_map uses .is_shard() which returns False for _StridedShard,
        so the shard dim is mapped to -1 (replicate) instead of the mesh dim.
        """
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        full = torch.randn(4, self.world_size * 2, 6, device=self.device_type)
        dt = distribute_tensor(full, mesh, [Shard(1)])
        dt_flat = dt.flatten(0, 1)

        self.assertIsInstance(dt_flat.placements[0], _StridedShard)
        shard_dim = dt_flat.placements[0].dim
        # The _StridedShard dim should be mapped to mesh dim 0, not -1.
        self.assertEqual(dt_flat._spec.dim_map[shard_dim], 0)

    @with_comms
    def test_num_shards_map_with_strided_shard(self):
        """DTensorSpec.num_shards_map must count _StridedShard placements per dim.

        num_shards_map uses .is_shard() which returns False for _StridedShard,
        so the shard dim shows 1 shard instead of world_size.
        """
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        full = torch.randn(4, self.world_size * 2, 6, device=self.device_type)
        dt = distribute_tensor(full, mesh, [Shard(1)])
        dt_flat = dt.flatten(0, 1)

        self.assertIsInstance(dt_flat.placements[0], _StridedShard)
        shard_dim = dt_flat.placements[0].dim
        self.assertEqual(dt_flat._spec.num_shards_map[shard_dim], self.world_size)


class TestStridedShardTrivialShard(DTensorTestBase):
    """Tests for is_trivial_shard with _StridedShard."""

    @property
    def world_size(self):
        return 4

    @with_comms
    def test_is_trivial_shard_strided_shard(self):
        """is_trivial_shard only checks isinstance(p, Shard), missing _StridedShard.

        A _StridedShard on a size-1 dim should also be considered trivial.
        """
        from torch.distributed.tensor._ops.strategy_validation import is_trivial_shard

        ss = _StridedShard(0, split_factor=2)
        self.assertTrue(
            is_trivial_shard(ss, (1, 8)),
            "_StridedShard on a size-1 dim should be trivial",
        )

    @with_comms
    def test_is_trivial_shard_strided_shard_non_trivial(self):
        """_StridedShard on a dim with size > 1 should NOT be trivial."""
        from torch.distributed.tensor._ops.strategy_validation import is_trivial_shard

        ss = _StridedShard(1, split_factor=2)
        self.assertFalse(is_trivial_shard(ss, (1, 8)))


class TestStridedShardScaledMM(DTensorTestBase):
    """Tests for _scaled_mm_scale_placement with _StridedShard."""

    @property
    def world_size(self):
        return 4

    @with_comms
    def test_scaled_mm_scale_placement_strided_shard_contracting(self):
        """_scaled_mm_scale_placement only checks isinstance(_, Shard) for the
        1D blockwise scale path. When data_placement is _StridedShard on the
        contracting dim, it should return None (reject), but instead falls
        through to 'return data_placement'.
        """
        from torch.distributed.tensor._ops._matrix_ops import _scaled_mm_scale_placement

        ss = _StridedShard(1, split_factor=2)
        # contracting_dim=1, 1D scale shape → should reject
        result = _scaled_mm_scale_placement(ss, contracting_dim=1, scale_shape=(4,))
        self.assertIsNone(
            result,
            f"Expected None for _StridedShard on contracting dim, got {result}",
        )

    @with_comms
    def test_scaled_mm_scale_placement_strided_shard_non_contracting(self):
        """_StridedShard on a non-contracting dim should map to Shard(0)
        for a 1D blockwise scale, same as regular Shard.
        """
        from torch.distributed.tensor._ops._matrix_ops import _scaled_mm_scale_placement

        ss = _StridedShard(0, split_factor=2)
        # contracting_dim=1, data on dim 0 → non-contracting, should map to Shard(0)
        result = _scaled_mm_scale_placement(ss, contracting_dim=1, scale_shape=(4,))
        self.assertIsInstance(
            result,
            Shard,
            f"Expected Shard(0) for _StridedShard on non-contracting dim, got {result}",
        )
        self.assertEqual(result.dim, 0)


if __name__ == "__main__":
    run_tests()
