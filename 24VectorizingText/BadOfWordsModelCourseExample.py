# Import the pandas library, which is used for creating and manipulating dataframes.
import pandas as pd
# Import CountVectorizer from scikit-learn to convert text into a matrix of token counts.
from sklearn.feature_extraction.text import CountVectorizer

# Define a list of strings (documents) that we will use as our sample data.
data = [' Most shark attacks occur about 10 feet from the beach since that is where the people are',
      'the efficiency with which he paired the socks in the drawer was quite admirable',
      'carol drank the blood as if she were a vampire',
      'giving directions that the mountains are to the west only works when you can see them',
      'the sign said there was road work ahead so he decided to speed up',
      'the gruff old man sat in the back of the bait shop grumbling to himself as he scooped out a handful of worms']

# --- Bag-of-Words (BoW) using CountVectorizer ---

# Create an instance of the CountVectorizer.
countvec = CountVectorizer()
# Learn the vocabulary from the data and transform the data into a document-term matrix (a sparse matrix).
countvec_fit = countvec.fit_transform(data)
# Create a pandas DataFrame from the vectorized data for easy viewing.
#   - countvec_fit.toarray(): Converts the sparse matrix of word counts into a dense NumPy array.
#   - columns=countvec.get_feature_names_out(): Retrieves the list of unique words (the vocabulary)
#     from the vectorizer to use as the column headers. This ensures each column is labeled
#     with the word it represents.
bag_of_words = pd.DataFrame(data=countvec_fit.toarray(), columns=countvec.get_feature_names_out())
# Print the resulting Bag-of-Words DataFrame. Each row is a document, and each column is a word from the vocabulary.
print(bag_of_words)

# --- Term Frequency-Inverse Document Frequency (TF-IDF) ---

# Import TfidfVectorizer from scikit-learn to convert text into a matrix of TF-IDF features.
from sklearn.feature_extraction.text import TfidfVectorizer
# Create an instance of the TfidfVectorizer.
tfidfvec = TfidfVectorizer()
# Learn the vocabulary and IDF from the data, and transform the data into a TF-IDF weighted document-term matrix.
tfidfvec_fit = tfidfvec.fit_transform(data)

print("Term Frequency-Inverse Document Frequency (TF-IDF)")
# Convert the resulting sparse matrix to a dense array and create a pandas DataFrame.
# The columns are the vocabulary words, and the values are the TF-IDF scores.
df_tfidf_bag = pd.DataFrame(tfidfvec_fit.toarray(), columns=tfidfvec.get_feature_names_out())
# Print the resulting TF-IDF DataFrame.
print(df_tfidf_bag)
