import torch
from torch.profiler import profile, ProfilerActivity

A = torch.randn(4096, 4096, device="cuda")
B = torch.randn(4096, 4096, device="cuda")
A16, B16 = A.half(), B.half()
A @ B; A16 @ B16                          # warm-up
torch.cuda.synchronize()

# record every GPU kernel PyTorch launches
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    A @ B                                 # FP32 matmul
    A16 @ B16                             # FP16 matmul
    torch.cuda.synchronize()

for e in prof.key_averages():
    if e.device_type.name == "CUDA":
        print(f"{e.device_time_total / 1000:6.2f} ms   {e.key[:110]}")
