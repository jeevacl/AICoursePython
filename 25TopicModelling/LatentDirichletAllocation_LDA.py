# --- 1. All your IMPORTS go at the top, in the global scope ---
# Import pandas for data manipulation and analysis.
import pandas as pd
# Import nltk for natural language processing tasks.
import nltk
# Import re for regular expressions, used for text cleaning.
import re
# Import stopwords from nltk.corpus to remove common words that don't add much meaning.
from nltk.corpus import stopwords
# Import word_tokenize from nltk.tokenize to break text into individual words.
from nltk.tokenize import word_tokenize
# Import PorterStemmer from nltk.stem for reducing words to their root form.
from nltk.stem import PorterStemmer
# Import gensim, a robust library for topic modeling.
import gensim
# Import corpora from gensim for creating a dictionary and document-term matrix.
from gensim import corpora
# Import LsiModel for Latent Semantic Indexing and CoherenceModel for evaluating topic models.
from gensim.models import LsiModel, CoherenceModel
# Import matplotlib.pyplot for plotting graphs, specifically for visualizing coherence scores.
import matplotlib.pyplot as plt

# --- 2. The MAIN GUARD ---
# All the code that *runs* your analysis must be inside this block.
if __name__ == '__main__':

    # --- 3. Download NLTK Data (safe inside the guard) ---
    # Downloads the 'stopwords' corpus from NLTK, which is necessary for text preprocessing.
    nltk.download('stopwords')

    # --- 4. Load and Inspect the Data ---
    # Loads the dataset from the specified CSV file into a pandas DataFrame.
    data = pd.read_csv("Practice/news_articles.csv")
    # Prints the first 5 rows of the DataFrame to get a quick overview of the data structure.
    print(data.head())
    # Prints a summary of the DataFrame, including column data types and non-null values.
    print(data.info())

    # --- 5. Text Preprocessing ---
    # Extracts the 'content' column from the DataFrame, which contains the news article texts.
    articles = data['content']
    # Prints the first few entries of the 'articles' Series to see the raw text.
    print(articles.head())
    # Converts all text to lowercase and removes punctuation using a regular expression.
    articles = articles.str.lower().apply(lambda x: re.sub(r"([^\w\s])", "", x))
    # Retrieves the list of English stopwords from NLTK.
    en_stopwords = stopwords.words('english')
    # Removes stopwords from each article. It splits the text into words, filters out stopwords, and then joins them back.
    articles = articles.apply(lambda x: ' '.join([word for word in x.split() if word not in en_stopwords]))
    # Tokenizes each article into a list of individual words.
    articles = articles.apply(lambda x: word_tokenize(x))
    # Initializes the Porter Stemmer.
    ps = PorterStemmer()
    # Applies stemming to each word in every article, reducing words to their base form.
    articles = articles.apply(lambda x: [ps.stem(word) for word in x])
    # Prints a message indicating the stemming process is complete.
    print("articles after stemming")
    # Prints the first few entries of the 'articles' Series after stemming to verify the changes.
    print(articles.head())

    # --- 6. Prepare Data for Models ---
    # Creates a dictionary from the preprocessed articles, mapping each unique word to an integer ID.
    dictionary = corpora.Dictionary(articles)
    # Converts each article into a bag-of-words representation (list of (token_id, count) tuples).
    doc_term_matrix = [dictionary.doc2bow(article) for article in articles]

    # --- 7. Build and Train the LDA Model ---
    # Sets the number of topics for the LDA model.
    num_topics = 2
    # Builds and trains the LDA (Latent Dirichlet Allocation) model using the document-term matrix.
    lda_model = gensim.models.ldamodel.LdaModel(doc_term_matrix, num_topics=num_topics, id2word=dictionary)
    # Prints a header for the discovered LDA topics.
    print("\nDiscovered Topics:")
    # Retrieves the top 5 words for each topic from the LDA model.
    topics = lda_model.print_topics(num_topics=num_topics, num_words=5)
    # Iterates through and prints each discovered LDA topic.
    for topic in topics:
        print(topic)

    # --- 8. Build and Train the LSA Model ---
    # Builds and trains the LSA (Latent Semantic Analysis) model using the document-term matrix.
    lsa_model = LsiModel(corpus=doc_term_matrix, id2word=dictionary, num_topics=num_topics)
    # Prints a header for the LSA topics.
    print("\nLSA Topics:")
    # Retrieves the top 5 words for each topic from the LSA model.
    lsa_topics = lsa_model.print_topics(num_topics=num_topics, num_words=5)
    # Iterates through and prints each discovered LSA topic.
    for topic in lsa_topics:
        print(topic)

    # --- 9. Evaluate LSA Model Coherence (THE PART THAT NEEDS THE GUARD) ---
    # Initializes an empty list to store coherence scores for different numbers of topics.
    coherence_values = []
    # Initializes an empty list to store LSA models trained with different numbers of topics.
    model_list = []
    # Sets the minimum number of topics to evaluate.
    min_topics = 2
    # Sets the maximum number of topics to evaluate (exclusive).
    max_topics = 11

    # Loops through a range of topic numbers to train LSA models and calculate their coherence.
    for num_topics_i in range(min_topics, max_topics):
        # Prints the current number of topics being evaluated.
        print(f"Training LSA model with {num_topics_i} topics...")
        # Trains an LSA model with the current number of topics.
        model = LsiModel(doc_term_matrix, id2word=dictionary, num_topics=num_topics_i)
        # Appends the trained model to the model_list.
        model_list.append(model)

        # This is the line that starts multiprocessing.
        # Creates a CoherenceModel object to evaluate the LSA model's coherence using 'c_v' measure.
        coherence_model = CoherenceModel(model=model, texts=articles, dictionary=dictionary, coherence='c_v')
        # Calculates the coherence score for the current model and appends it to the list.
        coherence_values.append(coherence_model.get_coherence())

    # --- 10. Plot the Results and Show Final Model ---
    # Defines the x-axis values for the plot, representing the number of topics.
    x = range(min_topics, max_topics)
    # Plots the coherence values against the number of topics.
    plt.plot(x, coherence_values)
    # Sets the label for the x-axis.
    plt.xlabel("Number of Topics")
    # Sets the label for the y-axis.
    plt.ylabel("Coherence Score")
    # Adds a legend to the plot.
    plt.legend(("coherence_values"), loc='best')

    # Example: choose the best number from your plot. This value would typically be chosen after inspecting the coherence plot.
    final_num_topics = 3
    # Trains the final LSA model using the chosen optimal number of topics.
    final_lsa_model = LsiModel(doc_term_matrix, id2word=dictionary, num_topics=final_num_topics)
    # Retrieves the top 5 words for each topic from the final LSA model.
    final_lsa_topics = final_lsa_model.print_topics(num_topics=final_num_topics, num_words=5)

    # Prints a header for the final LSA model topics.
    print("\n--- Final LSA Model Topics ---")
    # Iterates through and prints each topic from the final LSA model.
    for topic in final_lsa_topics:
        print(topic)
        # Prints an empty line for better readability between topics.
        print("\n")

    # Displays the plot showing the coherence scores.
    plt.show()