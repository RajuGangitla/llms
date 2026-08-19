w  = 0.5 
x = 2

for i in range(600):
    prediction  = x*w
    error = prediction - 4
    if abs(error) < 0.001:
        print("close enough!")
        break
    nudge  = 0.01 * error *x
    w = w - nudge
    print(7 * w)
    print(100 * w)