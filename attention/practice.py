# Exercise 1: Every word attends to every word
# Give all 3 words (cat, mat, sat) a query. Compute attention for EACH word.
# Output: 3 blended vectors, not 1.

# Exercise 2: Break it on purpose
# Remove the / sqrt(d) scaling and multiply all embeddings by 10.
# What happens to the softmax percentages?

# Exercise 3: Causal mask
# A word can't look at words AFTER it.
# cat sees only cat, mat sees cat+mat, sat sees all three.

# Exercise 4: Count the work
# Print how many dot products you computed for 3, 10, then 100 words.
# Notice it grows as n * n.
