import nltk
import pandas as pd
import matplotlib.pyplot as plt

# --- Step 1: Define the list of tokens ---
# This is the data from your notebook cell [3]
tokens = [
    'the', 'rise', 'of', 'artificial', 'intelligence', 'has', 'led', 'to', 'significant', 'advancements', 'in',
    'natural', 'language', 'processing', 'computer', 'vision', 'and', 'other', 'fields', 'machine', 'learning',
    'algorithms', 'are', 'becoming', 'more', 'sophisticated', 'enabling', 'computers', 'to', 'perform', 'complex',
    'tasks', 'that', 'were', 'once', 'thought', 'to', 'be', 'the', 'exclusive', 'domain', 'of', 'humans', 'with',
    'the', 'advent', 'of', 'deep', 'learning', 'neural', 'networks', 'have', 'become', 'even', 'more', 'powerful',
    'capable', 'of', 'processing', 'vast', 'amounts', 'of', 'data', 'and', 'learning', 'from', 'it', 'in',
    'ways', 'that', 'were', 'not', 'possible', 'before', 'as', 'a', 'result', 'ai', 'is', 'increasingly',
    'being', 'used', 'in', 'a', 'wide', 'range', 'of', 'industries', 'from', 'healthcare', 'to', 'finance',
    'to', 'transportation', 'and', 'its', 'impact', 'is', 'only', 'set', 'to', 'grow', 'in', 'the', 'years',
    'to', 'come'
]


# --- Step 2: Generate and count the unigrams ---
# This corresponds to your notebook cell [5]
# 1. nltk.ngrams(tokens, 1) creates the unigrams (sequences of 1 word).
# 2. pd.Series(...) converts them into a pandas Series for easy counting.
# 3. .value_counts() counts the occurrences of each unique unigram.
unigrams = pd.Series(nltk.ngrams(tokens, 1)).value_counts()

# .items() gives you both the index (the word tuple) and the value (the count)
for word_tuple, count in unigrams.items():
    # The word is inside the tuple, so we access it with word_tuple[0]
    word = word_tuple[0]
    print(f"Word: '{word}'  ---  Count: {count}")


# Print the top 10 most frequent unigrams
print("Top 10 most frequent unigrams:")
print(unigrams[:10])


# --- Step 3: Plot the results ---
# This is the code from your last notebook cell.
# We take the top 10 unigrams, sort them for better visualization, and create a horizontal bar plot.
# .sort_values() is added to make the chart ordered from least to most frequent.
plt.figure(figsize=(12, 8)) # This sets the figure size
unigrams[:10].sort_values().plot.barh(color='lightsalmon', width=0.9)

# Add a title to the plot
plt.title("10 Most Frequently Occurring Unigrams")

# Add labels for clarity
plt.xlabel("Frequency")
plt.ylabel("Unigrams")

# Ensure the layout is tight
plt.tight_layout()

# Display the plot
#plt.show()


print("Bibrams values:")
bigrams = pd.Series(nltk.ngrams(tokens, 2)).value_counts()
# Print the entire bigrams Series (optional, but good for checking)
print("--- Full Bigrams Series ---")
print(bigrams)
print("\n" + "="*30 + "\n") # Adding a separator for clarity

# --- THE CORRECTED LOOP ---
print("Bigrams values and count:")
# 1. Loop over bigrams.items(), NOT unigrams
for bigram_tuple, count in bigrams.items():
    # 2. Join the two words in the tuple with a space
    bigram_text = ' '.join(bigram_tuple)
    print(f"Bigram: '{bigram_text}'  ---  Count: {count}")




print("Trigram values:")
trigrams = pd.Series(nltk.ngrams(tokens, 3)).value_counts()
# Print the entire bigrams Series (optional, but good for checking)
print("--- Full Trigram Series ---")
print(trigrams)
print("\n" + "="*30 + "\n") # Adding a separator for clarity

# --- THE CORRECTED LOOP ---
print("Trigrams values and count:")
# 1. Loop over bigrams.items(), NOT unigrams
for trigram_tuple, count in trigrams.items():
    # 2. Join the two words in the tuple with a space
    trigram_text = ' '.join(trigram_tuple)
    print(f"trigram: '{trigram_text}'  ---  Count: {count}")