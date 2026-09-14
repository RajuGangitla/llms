import math

embeddings = {
    "cat": [0.9, 0.1],
    "mat": [0.1, 0.8],
    "sat": [0.5, 0.5]
}

scores = {"cat": 0.5, "mat": 0.3, "sat": 0.2}

result1 = embeddings["cat"][0]*scores["cat"]+embeddings["mat"][0]*scores["mat"]+embeddings["sat"][0]*scores["sat"]
result2 = embeddings["cat"][1]*scores["cat"]+embeddings["mat"][1]*scores["mat"]+embeddings["sat"][1]*scores["sat"]


query_it = [0.8, 0.2]

keys = {
    "cat": [0.9, 0.1],
    "mat": [0.1, 0.7],
    "sat": [0.3, 0.4]
}

result = {}
total = 0
for key in keys:
    score = (keys[key][0]*query_it[0] + keys[key][1]*query_it[1]) / math.sqrt(2)
    total += math.exp(score)

for key in keys:
    score = (keys[key][0]*query_it[0] + keys[key][1]*query_it[1]) / math.sqrt(2)
    print(key, math.exp(score) / total)
    result[key] = math.exp(score) / total


final_result = result["cat"]*embeddings["cat"][0] + result["mat"]*embeddings["mat"][0] + result["sat"]*embeddings["sat"][0], result["cat"]*embeddings["cat"][1] + result["mat"]*embeddings["mat"][1] + result["sat"]*embeddings["sat"][1]
print(final_result)