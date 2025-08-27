# Cell 1: Importing necessary libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import re
import pandas as pd

# Cell 2: Reading the CSV file into a pandas DataFrame
# Make sure the 'tripadvisor_hotel_reviews.csv' file is in the correct directory.
data = pd.read_csv("tripadvisor_hotel_reviews/tripadvisor_hotel_reviews.csv")

# Cell 3: Displaying information about the DataFrame
print("--- DataFrame Info ---")
data.info()
print("\n" + "="*30 + "\n") # Adding a separator

# Displaying the first 5 rows of the DataFrame
# This line is now correctly aligned with zero indentation.
print("--- First 5 Rows of Data ---")
print(data.head())

print(data['Review'][0])

data['Review_lowar_Case'] = data['Review'].str.lower()

print(data['Review_lowar_Case'][0])