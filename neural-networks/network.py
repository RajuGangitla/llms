from re import S


data = [(1, 1), (2, 4), (3, 9), (4, 16)]

# layer 1 weights (random)
w1 = 0.3
b1 = 0.1
w2 = -0.5
b2 = 0.2

# layer 2 weights (random)
w3 = 0.7
b3 = 0.1
w4 = -0.3
lr = 0.001
x = 3
expected = 9


for step in range(1000):
    total_error = 0
    for x, expected in data:
        n1 = max(0, x * w1 + b1)
        n2 = max(0, x * w2 + b2)
        output = n1 * w3 + n2 * w4 + b3

        error = output - expected
        total_error += error ** 2

        w3 = w3 - lr * error * n1
        w4 = w4 - lr * error * n2
        b3 = b3 - lr * error
        w1 = w1 - lr * error * w3 * x
        b1 = b1 - lr * error * w3
        w2 = w2 - lr * error * w4 * x
        b2 = b2 - lr * error * w4

    if step % 100 == 0:
        print(f"step {step} | error={total_error:.4f}")

print("\npredictions:")
for x, expected in data:
    n1 = max(0, x * w1 + b1)
    n2 = max(0, x * w2 + b2)
    output = n1 * w3 + n2 * w4 + b3
    print(f"input={x} predicted={output:.2f} expected={expected}")