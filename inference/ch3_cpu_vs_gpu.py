import torch, time
torch.set_num_threads(1)

A = torch.randn(4096, 4096)
B = torch.randn(4096, 4096)

t = time.time()
A @ B
print("CPU:", round((time.time() - t) * 1000), "ms")

A = A.to("cuda")
B = B.to("cuda")
A @ B                          # warm-up

torch.cuda.synchronize()
t = time.time()
A @ B
torch.cuda.synchronize()
print("GPU:", round((time.time() - t) * 1000), "ms")

A16 = A.half()                 # same matrices, 16 bits instead of 32
B16 = B.half()
A16 @ B16                      # warm-up

torch.cuda.synchronize()
t = time.time()
A16 @ B16
torch.cuda.synchronize()
print("GPU FP16:", round((time.time() - t) * 1000), "ms")
