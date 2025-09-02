# --- 1. All your IMPORTS go at the top, in the global scope ---
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import gensim
from gensim import corpora
from gensim.models import LsiModel, CoherenceModel
import matplotlib.pyplot as plt

# --- 2. The MAIN GUARD ---
# All the code that *runs* your analysis must be inside this block.
if __name__ == '__main__':

    # --- 3. Download NLTK Data (safe inside the guard) ---
    nltk.download('stopwords')

    # --- 4. Load and Inspect the Data ---
    data = pd.read_csv("Practice/news_articles.csv")
    print(data.head())
    print(data.info())

    # --- 5. Text Preprocessing ---
    articles = data['content']
    print(articles.head())
    articles = articles.str.lower().apply(lambda x: re.sub(r"([^\w\s])", "", x))
    en_stopwords = stopwords.words('english')
    articles = articles.apply(lambda x: ' '.join([word for word in x.split() if word not in en_stopwords]))
    articles = articles.apply(lambda x: word_tokenize(x))
    ps = PorterStemmer()
    articles = articles.apply(lambda x: [ps.stem(word) for word in x])
    print("articles after stemming")
    print(articles.head())

    # --- 6. Prepare Data for Models ---
    dictionary = corpora.Dictionary(articles)
    doc_term_matrix = [dictionary.doc2bow(article) for article in articles]

    # --- 7. Build and Train the LDA Model ---
    num_topics = 2
    lda_model = gensim.models.ldamodel.LdaModel(doc_term_matrix, num_topics=num_topics, id2word=dictionary)
    print("\nDiscovered Topics:")
    topics = lda_model.print_topics(num_topics=num_topics, num_words=5)
    for topic in topics:
        print(topic)

    # --- 8. Build and Train the LSA Model ---
    lsa_model = LsiModel(corpus=doc_term_matrix, id2word=dictionary, num_topics=num_topics)
    print("\nLSA Topics:")
    lsa_topics = lsa_model.print_topics(num_topics=num_topics, num_words=5)
    for topic in lsa_topics:
        print(topic)

    # --- 9. Evaluate LSA Model Coherence (THE PART THAT NEEDS THE GUARD) ---
    coherence_values = []
    model_list = []
    min_topics = 2
    max_topics = 11

    for num_topics_i in range(min_topics, max_topics):
        print(f"Training LSA model with {num_topics_i} topics...")
        model = LsiModel(doc_term_matrix, id2word=dictionary, num_topics=num_topics_i)
        model_list.append(model)

        # This is the line that starts multiprocessing
        coherence_model = CoherenceModel(model=model, texts=articles, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherence_model.get_coherence())

    # --- 10. Plot the Results and Show Final Model ---
    x = range(min_topics, max_topics)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score")
    plt.legend(("coherence_values"), loc='best')

    final_num_topics = 3  # Example: choose the best number from your plot
    final_lsa_model = LsiModel(doc_term_matrix, id2word=dictionary, num_topics=final_num_topics)
    final_lsa_topics = final_lsa_model.print_topics(num_topics=final_num_topics, num_words=5)

    print("\n--- Final LSA Model Topics ---")
    for topic in final_lsa_topics:
        print(topic)
        print("\n")

    plt.show()