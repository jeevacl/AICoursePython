# --- Data Manipulation & Visualization ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- NLP Libraries (spaCy, NLTK, etc.) ---
import spacy
from spacy import displacy
from spacy import tokenizer
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- Topic Modeling (Gensim) ---
import gensim
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import LsiModel, TfidfModel

# --- Machine Learning (Scikit-Learn) ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report


# --- Set Plot Options ---
# This sets the default size for all plots created with matplotlib
plt.rcParams['figure.figsize'] = (12, 8)

# This defines a custom color variable to be used later in plotting
default_plot_colour = "#00bfbf"
data = pd.read_csv("PracticeData/fake_news_data.csv")
print("data head")
print(data.head())
print("data info")
print(data.info())

data["fake_or_factual"].value_counts().plot(kind="bar", color=default_plot_colour)
plt.title("Count Of Article Classification.")
#plt.show()

#-----------------------------------------#
# POS tagging
#-----------------------------------------#

# Run this comment in terminal: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

fake_news = data[data['fake_or_factual'] == "Fake News"]
fact_news = data[data['fake_or_factual'] == "Factual News"]

fake_spacydocs = list(nlp.pipe(fake_news['text']))
fact_spacydocs = list(nlp.pipe(fact_news['text']))

