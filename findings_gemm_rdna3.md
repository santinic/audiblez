# GEMM backends on gfx1101 (RX 7800 XT) — measured on-box, torch 2.12.1+rocm7.2

Validator: PT 2.12.1, HIP 702, HIPBLASLT 100202, ROCBLAS 5.2.0, GCN gfx1101

fp16 4096x4096x4096 matmul, 50 iters, 10 warmup:
- default (rocBLAS):           2.524 ms / 54.4 TFLOPS
- TORCH_BLAS_PREFER_HIPBLASLT=1: 3.044 ms / 45.1 TFLOPS  (SLOWER, ~17% regression)
- TORCH_BLAS_PREFER_HIPBLASLT=0: 2.533 ms / 54.3 TFLOPS  (== default)
- PYTORCH_TUNABLEOP_ENABLED=1 + TUNING=1: 2.493 ms / 55.1 TFLOPS (~1%, noise)

TunableOp winning solution in tunableop_results0.csv = "Default" (rocBLAS), i.e. tuning
searched hipBLASLt and could NOT beat rocBLAS default on gfx1101.

torch.backends.cuda.preferred_blas_library() default = _BlasBackend.Cublas (=rocBLAS path).

Conclusion: on gfx1101, rocBLAS is the default AND the fast path. hipBLASLt has gfx110x
kernels but they are NOT competitive on Navi32; forcing them hurts. TunableOp is flat for
medium GEMMs because rocBLAS already wins.
