# sentences = [
#     "hello world",
#     "hello there",
#     "hello friend",
#     "good morning",
#     "good night"
# ]

# d = {}

# for word in sentences:
#     parts = word.split(" ")
#     if parts[0] not in d:
#         d[parts[0]] = []
#     d[parts[0]].append(parts[1])


# input_word = "hello"

# if input_word in d:
#     print(d[input_word])
# else:
#     print("Word not found")


sentences = [
    "hello world",
    "hello world",
    "hello world",
    "hello there",
    "hello friend",
    "good morning",
    "good morning",
    "good night"
]

d ={}
for sentence in sentences:
    parts = sentence.split(" ")
    if parts[0] not in d:
        d[parts[0]] = {}
    if parts[1] not in d[parts[0]]:
                d[parts[0]][parts[1]] = 1
    else:
        d[parts[0]][parts[1]] = d[parts[0]][parts[1]] + 1
        
input_word = "hello"

best_count = 0
best_word = ""
for word in d[input_word]:
    if d[input_word][word] > best_count:
        best_count = d[input_word][word]
        best_word = word

print(best_word)