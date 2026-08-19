x1 = 2
x2 = 3

w1 = -1.0
w2 = 0.5

bias = 0.1

result = x1 * w1 + x2 * w2 + bias
print("before activation:", result)

output = max(0, result)
print("after activation:", output)
