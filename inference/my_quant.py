import torch

w = torch.tensor([100.0, -7.59, 3.08, 3.02, 0.25, -1.40]).reshape(2, 3)   # 2 blocks of 3
coin = w.abs().amax(dim=1, keepdim=True) / 127                             # one coin per block

stored = (w / coin).round()

back = stored * coin

error = w - back

print("original :", w)
print("coin     :", coin)
print("stored   :", stored)
print("read back:", back)
print("error    :", error)
