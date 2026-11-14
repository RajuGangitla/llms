import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16)

# one real weight matrix: the MLP down_proj of the last block (block 27)
W = model.model.layers[27].mlp.down_proj.weight.data.float()

print("shape      :", W.shape)
print("numbers    :", W.numel())
print("mean |w|   :", W.abs().mean().item())
print("absmax     :", W.abs().max().item())
print("outlier is :", (W.abs().max() / W.abs().mean()).item(), "x the mean")


def quantize(W, bits, block):
    qmax = 2 ** (bits - 1) - 1                                        # INT8 → 127, INT4 → 7
    g = W.reshape(1, -1) if block is None else W.reshape(-1, block)   # per-tensor: 1 group, per-block: rows of `block`
    scale = g.abs().amax(dim=1, keepdim=True) / qmax                  # absmax scale for each group
    q = (g / scale).round().clamp(-qmax, qmax)                        # quantize (RTN)
    return (q * scale).reshape(W.shape)                               # dequantize, back to original shape


print()
for bits, block in [(4, 8), (4, 16), (4, 32), (4, 64), (4, 128), (4, 1024), (4, None)]:
    W_hat = quantize(W, bits, block)
    zeros = (W_hat == 0).float().mean().item() * 100                  # % of weights that became 0
    error = (W - W_hat).abs().mean().item()                           # mean quantization error
    real_bits = bits + (0 if block is None else 16 / block)           # 4 bits + one 16-bit scale per block
    name = f"INT{bits} " + ("per-tensor" if block is None else f"block {block}")
    print(f"{name:18} zeros: {zeros:5.1f} %   mean error: {error:.3f}")