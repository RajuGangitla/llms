import random

a = ["king", "queen", "man", "woman", "apple"]

d = {}

for i in a:
    d[i] = [random.random(), random.random(), random.random()]

print(d)


def similarity_search(w1, w2):
    vector1 = d[w1]
    vector2 = d[w2]

    distance = sum((a - b) ** 2 for a, b in zip(vector1, vector2)) ** 0.5

    return distance


print(similarity_search("king", "queen"))