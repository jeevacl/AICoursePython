import re
from collections import Counter


# --- Step 1: Tokenization ---
# This function prepares the text for processing.
def tokenize(text):
    # Convert the entire text to lowercase to ensure "The" and "the" are treated as the same word.
    text = text.lower()
    # Use a regular expression to remove any characters that are not lowercase letters, numbers, or whitespace.
    # This cleans up punctuation like periods and commas.
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Split the cleaned text into a list of individual words.
    words = text.split()
    return words


# --- Step 2: Vocabulary Building ---
# This function creates a master list of all unique words from a collection of documents.
def build_vocabulary(documents):
    # A 'set' is used to automatically store only unique words. No duplicates allowed.
    vocabulary = set()
    # Loop through each document in our collection.
    for doc in documents:
        # Tokenize the document to get its words.
        words = tokenize(doc)
        # Add all the words from the current document to our vocabulary set.
        vocabulary.update(words)
    # Convert the set to a list and sort it alphabetically. This provides a consistent order.
    return sorted(list(vocabulary))


# --- Step 3: Vectorization (Creating the Bag-of-Words) ---
# This function converts a single piece of text into a numerical vector.
def create_bag_of_words(text, vocabulary):
    # First, tokenize the input text.
    words = tokenize(text)
    # Use collections.Counter to count the occurrences of each word in the tokenized text.
    # e.g., for "the dog the", it would produce {'the': 2, 'dog': 1}
    word_counts = Counter(words)
    # Create the vector. The vector will have the same length as the vocabulary.
    # For each word in our master vocabulary...
    # ...get its count from the current text. If the word isn't in the text, default to 0.
    bag_of_words = [word_counts.get(word, 0) for word in vocabulary]
    return bag_of_words


# --- Main execution block ---
if __name__ == "__main__":
    # Our collection of documents (the "corpus").
    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "A quick brown cat sleeps under the table.",
        "The dog barks loudly."
    ]

    print("--- Bag of Words Model Example ---")

    # First, build a single vocabulary from all documents.
    vocabulary = build_vocabulary(documents)
    print("\nVocabulary:", vocabulary)
    # The output is: ['a', 'barks', 'brown', 'cat', 'dog', 'fox', 'jumps', 'lazy', 'loudly', 'over', 'quick', 'sleeps', 'table', 'the', 'under']

    # Now, create a Bag-of-Words vector for each document.
    print("\nBag of Words for each document:")
    for i, doc in enumerate(documents):
        # Convert the document into its vector representation.
        bow = create_bag_of_words(doc, vocabulary)
        print(f"Document {i + 1}: '{doc}'")
        print(f"Bag of Words: {bow}")
        # The vector's values correspond to the counts of the words in the vocabulary list above.
        # For Document 1, the vector [0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 2, 0] means:
        # 'a': 0, 'barks': 0, 'brown': 1, 'cat': 0, 'dog': 1, 'fox': 1, ... 'the': 2, ...

    # --- Example with a new sentence ---
    # Let's see how the model represents a new sentence it hasn't seen before.
    new_sentence = "The brown dog is quick."

    # We use the SAME vocabulary we built earlier.
    # The word "is" is not in our vocabulary, so it will be ignored.
    new_sentence_bow = create_bag_of_words(new_sentence, vocabulary)
    print(f"\nBag of Words for new sentence: '{new_sentence}'")
    print(f"Bag of Words: {new_sentence_bow}")
    # The vector [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0] corresponds to:
    # 'brown': 1, 'dog': 1, 'quick': 1, 'the': 1. All other counts are 0.
