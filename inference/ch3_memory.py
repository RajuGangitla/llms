import torch, time

for mb in [8, 16, 64, 256, 1024, 4096]:
    n = mb * 1024 * 1024 // 4               # how many 4-byte numbers fit in this many MB
    x = torch.ones(n, device="cuda", dtype=torch.float32)
    x.sum()                                 # warm-up: puts it in the cache if it fits

    reps = max(1, 8192 // mb)               # read about 8 GB in total
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(reps):
        x.sum()                             # .sum() has to read every number
    torch.cuda.synchronize()
    dt = time.time() - t

    print(f"{mb:5} MB  {mb * reps / 1024 / dt:7.0f} GB/s")
    del x
    torch.cuda.empty_cache()
