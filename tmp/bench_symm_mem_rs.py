#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.distributed._symmetric_memory as symm_mem


@dataclass
class BenchResult:
    case: str
    mode: str
    world_size: int
    m: int
    k: int
    n: int
    scatter_dim: int
    fused_ms: float
    baseline_ms: float
    speedup: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark symm_mem fused RS+GEMM against unfused baselines."
    )
    parser.add_argument(
        "--mode",
        choices=["bf16", "fp8"],
        default="bf16",
        help="bf16 uses fused_matmul_reduce_scatter; fp8 uses fused_scaled_matmul_reduce_scatter.",
    )
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--k", type=int, default=7168)
    parser.add_argument("--n", type=int, default=16384)
    parser.add_argument("--scatter-dim", type=int, choices=[0, 1], default=0)
    parser.add_argument(
        "--m-list",
        type=int,
        nargs="+",
        default=[],
        help="Optional sweep over multiple M values. If set, overrides --m.",
    )
    parser.add_argument(
        "--scatter-dim-list",
        type=int,
        nargs="+",
        choices=[0, 1],
        default=[],
        help="Optional sweep over multiple scatter_dim values. If set, overrides --scatter-dim.",
    )
    parser.add_argument("--reduce-op", choices=["sum", "avg"], default="sum")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Collect a short torch.profiler trace for both fused and baseline.",
    )
    parser.add_argument(
        "--profile-iters",
        type=int,
        default=10,
        help="Number of iterations to capture in profiler mode.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional JSON output path written by rank 0.",
    )
    return parser.parse_args()


def init_dist() -> tuple[int, int, str]:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl", device_id=torch.device(f"cuda:{local_rank}")
        )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, dist.group.WORLD.group_name


def check_args(args: argparse.Namespace, world_size: int) -> None:
    if args.scatter_dim == 0 and args.m % world_size != 0:
        raise ValueError(
            f"m={args.m} must be divisible by world_size={world_size} when scatter_dim=0"
        )
    if args.scatter_dim == 1 and args.n % world_size != 0:
        raise ValueError(
            f"n={args.n} must be divisible by world_size={world_size} when scatter_dim=1"
        )


def make_inputs_bf16(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    a = torch.randn((args.m, args.k), device=device, dtype=torch.bfloat16)
    b = torch.randn((args.k, args.n), device=device, dtype=torch.bfloat16)
    return a, b


def make_inputs_fp8(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("This PyTorch build does not expose torch.float8_e4m3fn")
    a = torch.randn((args.m, args.k), device=device, dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    b = torch.randn((args.k, args.n), device=device, dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    a_scale = torch.tensor([1.0], device=device, dtype=torch.float32)
    b_scale = torch.tensor([1.0], device=device, dtype=torch.float32)
    return a, b, a_scale, b_scale


def fused_bf16(
    a: torch.Tensor,
    b: torch.Tensor,
    reduce_op: str,
    scatter_dim: int,
    group_name: str,
) -> torch.Tensor:
    return torch.ops.symm_mem.fused_matmul_reduce_scatter(
        a, b, reduce_op, scatter_dim, group_name
    )


def baseline_bf16(
    a: torch.Tensor,
    b: torch.Tensor,
    reduce_op: str,
    scatter_dim: int,
    group_name: str,
) -> torch.Tensor:
    out = torch.matmul(a, b)
    out = funcol.reduce_scatter_tensor(out, reduce_op, scatter_dim, group_name)
    return funcol.wait_tensor(out)


def fused_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    reduce_op: str,
    scatter_dim: int,
    group_name: str,
) -> torch.Tensor:
    return torch.ops.symm_mem.fused_scaled_matmul_reduce_scatter(
        a,
        b,
        a_scale,
        b_scale,
        reduce_op,
        scatter_dim,
        scatter_dim,
        group_name,
        [a.shape[0], b.shape[1]],
        None,
        None,
        torch.bfloat16,
        False,
    )


def baseline_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    reduce_op: str,
    scatter_dim: int,
    group_name: str,
) -> torch.Tensor:
    out = torch._scaled_mm(
        a,
        b,
        a_scale,
        b_scale,
        None,
        None,
        torch.bfloat16,
        False,
    )
    out = funcol.reduce_scatter_tensor(out, reduce_op, scatter_dim, group_name)
    return funcol.wait_tensor(out)


def time_callable(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def maybe_profile(name: str, fn, enabled: bool, iters: int, rank: int) -> None:
    if not enabled:
        return
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(activities=activities) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    trace = f"/tmp/{name}_rank{rank}.json"
    prof.export_chrome_trace(trace)
    if rank == 0:
        print(f"[profile] wrote {trace}")


def make_case_args(
    args: argparse.Namespace, m: int, scatter_dim: int
) -> argparse.Namespace:
    return argparse.Namespace(**{**vars(args), "m": m, "scatter_dim": scatter_dim})


def run_case(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    group_name: str,
) -> BenchResult:
    check_args(args, world_size)
    device = torch.device("cuda", torch.cuda.current_device())

    if args.mode == "bf16":
        a, b = make_inputs_bf16(args, device)

        def fused() -> torch.Tensor:
            return fused_bf16(a, b, args.reduce_op, args.scatter_dim, group_name)

        def baseline() -> torch.Tensor:
            return baseline_bf16(a, b, args.reduce_op, args.scatter_dim, group_name)

        case = "rs+gemm"
    else:
        a, b, a_scale, b_scale = make_inputs_fp8(args, device)

        def fused() -> torch.Tensor:
            return fused_fp8(
                a, b, a_scale, b_scale, args.reduce_op, args.scatter_dim, group_name
            )

        def baseline() -> torch.Tensor:
            return baseline_fp8(
                a, b, a_scale, b_scale, args.reduce_op, args.scatter_dim, group_name
            )

        case = "rs+scaled_mm"

    fused_ms = time_callable(fused, args.warmup, args.iters)
    baseline_ms = time_callable(baseline, args.warmup, args.iters)

    maybe_profile(
        f"symm_mem_fused_{args.mode}_m{args.m}_sd{args.scatter_dim}",
        fused,
        args.profile,
        args.profile_iters,
        rank,
    )
    maybe_profile(
        f"symm_mem_baseline_{args.mode}_m{args.m}_sd{args.scatter_dim}",
        baseline,
        args.profile,
        args.profile_iters,
        rank,
    )

    return BenchResult(
        case=case,
        mode=args.mode,
        world_size=world_size,
        m=args.m,
        k=args.k,
        n=args.n,
        scatter_dim=args.scatter_dim,
        fused_ms=fused_ms,
        baseline_ms=baseline_ms,
        speedup=baseline_ms / fused_ms,
    )


def print_summary(results: list[BenchResult]) -> None:
    print("mode  m      scatter_dim  fused_ms   baseline_ms  speedup")
    for r in results:
        print(
            f"{r.mode:<5} {r.m:<6} {r.scatter_dim:<12} "
            f"{r.fused_ms:<10.6f} {r.baseline_ms:<12.6f} {r.speedup:.4f}"
        )


def main() -> None:
    args = parse_args()
    rank, world_size, group_name = init_dist()

    m_values = args.m_list if args.m_list else [args.m]
    scatter_dims = (
        args.scatter_dim_list if args.scatter_dim_list else [args.scatter_dim]
    )

    results: list[BenchResult] = []
    for m in m_values:
        for scatter_dim in scatter_dims:
            case_args = make_case_args(args, m, scatter_dim)
            result = run_case(case_args, rank, world_size, group_name)
            results.append(result)
            if rank == 0:
                print(json.dumps(asdict(result), indent=2))

    if rank == 0:
        print_summary(results)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump([asdict(r) for r in results], f, indent=2)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
