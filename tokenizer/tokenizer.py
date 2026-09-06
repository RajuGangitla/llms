a = "Hello"

tokens =  []
for char in a:
    tokens.append(ord(char))

res = ""

for i in tokens :
    res+=chr(i)


a = "hello world hello there"

tokens = a.split(" ")
d= {}

for token in tokens:
    if token not in d:
        d[token] = len(d)

print(d)


result = []

for token in tokens:
    result.append(d[token])

print(result)