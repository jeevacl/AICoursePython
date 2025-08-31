# This file uses environment 3.12
# --- 1. Import Necessary Libraries ---

# pandas is used for data manipulation and analysis, particularly for handling the CSV data.
import pandas as pd
# nltk (Natural Language Toolkit) is a suite of libraries for symbolic and statistical NLP.
import nltk
# re is the regular expression library, used for text cleaning and pattern matching.
import re
# stopwords contains a list of common words (like 'the', 'a', 'in') that are often removed from text.
from nltk.corpus import stopwords
# word_tokenize is used to split text into individual words (tokens).
from nltk.tokenize import word_tokenize
# PorterStemmer is used for stemming, which reduces words to their root form (e.g., "running" -> "run").
from nltk.stem import PorterStemmer
# gensim is a robust open-source library for unsupervised topic modeling and natural language processing.
import gensim
# corpora is a submodule of gensim used to create a dictionary and corpus for topic modeling.
from gensim import corpora

# Download necessary NLTK data. 'stopwords' for filtering common words.
# Note: You only need to run this once. It can be commented out after the first run.
nltk.download('stopwords')

# --- 2. Load and Inspect the Data ---

# Load the dataset from a CSV file into a pandas DataFrame.
data = pd.read_csv("Practice/news_articles.csv")
# Print the first 5 rows of the DataFrame to get a quick look at the data structure.
# Sample Output:
#      author         date    ...                                            content   title
# 0  Blood-Red ...  2016-12-07    ...  When it comes to big-money transfers, Liverpoo...   NaN
# 1  Blood-Red ...  2016-12-07    ...  When it comes to big-money transfers, Liverpoo...   NaN
# 2  Chronicle ...  2016-12-07    ...  A black box flight recorder has been found nea...   NaN
print(data.head())
# Print a concise summary of the DataFrame, including data types and non-null values.
# Sample Output:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 2748 entries, 0 to 2747
# Data columns (total 6 columns):
#  #   Column      Non-Null Count  Dtype
# ---  ------      --------------  -----
#  0   author      2748 non-null   object
# ...
#  4   content     2748 non-null   object
#  5   title       2748 non-null   object
print(data.info())

# --- 3. Text Preprocessing ---
# This is a crucial step to clean and standardize the text data for the model.

# Select only the 'content' column which contains the text of the news articles.
articles = data['content']
# Print the first few articles to see the raw text.
# Sample Output:
# 0    When it comes to big-money transfers, Liverpoo...
# 1    When it comes to big-money transfers, Liverpoo...
print(articles.head())

# Step 3a: Convert to lowercase and remove punctuation.
# .str.lower() converts all text to lowercase to ensure 'The' and 'the' are treated as the same word.
# .apply() with a lambda function processes each article.
# re.sub(r"([^\w\s])", "", x) removes any character that is NOT a word character (\w) or whitespace (\s).
articles = articles.str.lower().apply(lambda x: re.sub(r"([^\w\s])", "", x))
# Sample Output for one article: 'when it comes to bigmoney transfers liverpool have had a mixed record'

# Step 3b: Stop word removal.
# Get the list of standard English stop words from NLTK.
en_stopwords = stopwords.words('english')
# For each article, split it into words, and join them back together, keeping only the words NOT in the stop words list.
articles = articles.apply(lambda x: ' '.join([word for word in x.split() if word not in en_stopwords]))
# Sample Output for one article: 'comes bigmoney transfers liverpool mixed record'

# Step 3c: Tokenization.
# Split each article string into a list of individual words (tokens).
articles = articles.apply(lambda x: word_tokenize(x))
# Sample output for one article: ['new', 'york', 'citi', 'mayor', 'bill', 'de', 'blasio', 'said', ...]

# Step 3d: Stemming.
# This reduces words to their root form to group related words. It's done for speed and to reduce vocabulary size.
ps = PorterStemmer()
# For each list of tokens (article), apply the stemming function to each word.
articles = articles.apply(lambda x: [ps.stem(word) for word in x])
# Sample Output:
# 0    [come, bigmoney, transfer, liverpool, mix, rec...
# 1    [come, bigmoney, transfer, liverpool, mix, rec...
# 2    [black, box, flight, record, found, near, cras...
# ...
print("articles after stemming")
print(articles.head())

# --- 4. Prepare Data for LDA Model (Gensim) ---

# Step 4a: Create a Dictionary.
# A gensim Dictionary maps each unique word (token) to a unique integer ID.
dictionary = corpora.Dictionary(articles)
# The dictionary contains mappings like: {0: 'bigmoney', 1: 'come', 2: 'liverpool', ...}

# Step 4b: Create a Corpus (Document-Term Matrix).
# doc2bow (document to bag-of-words) converts each tokenized article into a list of (word_id, word_count) tuples.
# This is the numerical representation of the text that the LDA model requires.
doc_term_matrix = [dictionary.doc2bow(article) for article in articles]
# Sample Output for one article: [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), ...]
# This means word with ID 0 appears 1 time, word with ID 1 appears 1 time, etc.

# --- 5. Build and Train the LDA Model ---

# Define the number of topics the model should discover.
num_topics = 2
# Create an instance of the LdaModel.
# - corpus: The document-term matrix (our bag-of-words).
# - num_topics: The number of topics to be extracted.
# - id2word: The dictionary to map word IDs back to words for interpreting the topics.
lda_model = gensim.models.ldamodel.LdaModel(doc_term_matrix, num_topics, id2word=dictionary)

# --- 6. View the Discovered Topics ---

# Print the topics discovered by the model.
# .print_topics() returns a list of strings, where each string represents a topic
# and is composed of the most influential words for that topic.
print("\nDiscovered Topics:")
topics = lda_model.print_topics(num_topics=num_topics, num_words=5)
# Sample Output:
# (0, '0.008*"said" + 0.005*"year" + 0.004*"new" + 0.004*"peopl" + 0.003*"like"')
# (1, '0.007*"said" + 0.004*"trump" + 0.004*"year" + 0.003*"new" + 0.003*"state"')
for topic in topics:
    print(topic)


#--------------------Latent Semantic Analysis (LSA) ------------------------------------------------------#
#
# ### What is Latent Semantic Analysis (LSA)?
#
# Latent Semantic Analysis (LSA), also known as Latent Semantic Indexing (LSI), is another powerful technique
# used for topic modeling and information retrieval. Its core idea is to find the hidden (or "latent") semantic
# structure in a collection of texts. It achieves this by analyzing the statistical relationships between words
# and documents.
#
# ### How does LSA work?
#
# LSA is based on a linear algebra technique called Singular Value Decomposition (SVD). The process is as follows:
#
# 1.  **Create a Document-Term Matrix**: First, the text corpus is converted into a numerical matrix.
#     While a simple word count (Bag-of-Words) can be used, LSA performs much better with a
#     **TF-IDF (Term Frequency-Inverse Document Frequency)** matrix. TF-IDF gives higher weights to words
#     that are important to a specific document but are not common across all documents.
#
# 2.  **Apply Singular Value Decomposition (SVD)**: SVD is used to decompose the TF-IDF matrix into three
#     separate matrices. This decomposition essentially identifies the underlying patterns and concepts
#     in the data.
#
# 3.  **Dimensionality Reduction**: The key step in LSA is to reduce the dimensionality of these matrices.
#     By keeping only the first 'k' most significant dimensions (where 'k' is the number of topics you want
#     to find), the model compresses the information. This forces the algorithm to learn the relationships
#     between words that occur in similar contexts. For example, it can learn that "car", "automobile", and
#     "vehicle" are related, even if they don't appear in the same document.
#
# 4.  **Topics as Dimensions**: In the resulting low-dimensional space, each dimension represents a "topic".
#     A topic is essentially a collection of related words, and each document is represented as a mixture
#     of these topics.
#
# ### LSA vs. LDA
#
# - **Mathematical Foundation**: LSA is based on deterministic linear algebra (SVD), while LDA is based on
#   probabilistic models.
# - **Interpretability**: LDA topics are often easier for humans to interpret because they are probability
#   distributions over words. LSA topics are mathematical dimensions and can sometimes be less intuitive.
# - **Use Case**: LSA is excellent for tasks like finding similar documents (information retrieval) and as a
#   general-purpose dimensionality reduction technique for text. LDA is often preferred when the goal is to
#   discover human-readable thematic structures in a corpus.


