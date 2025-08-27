# Import necessary libraries
import pandas as pd
import numpy as np
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import matplotlib.pyplot as plt # Import the plotting library

# Load the dataset from a CSV file
data = pd.read_csv("Practice/book_reviews_sample.csv")

# --- Initial Data Exploration ---
# Display the first few rows of the dataframe to understand its structure.
print("Data As It Is")
print(data.head())
# Display a concise summary of the dataframe, including data types and non-null values.
print(data.info())
print("\n" + "="*30 + "\n")

# Print the raw text of the first review for a closer look.
print("First Row Data As It Is")
print(data['reviewText'][0])


# --- Data Cleaning ---
# Create a new column 'reviewText_Clean' with cleaned text.
# The cleaning process involves converting the text to lowercase and removing punctuation and special characters.
print("After clean the data")
data['reviewText_Clean'] = data.apply(lambda row: re.sub('[^\w\s]', ' ', row['reviewText'].lower()), axis=1)
# Display the head of the dataframe to see the new cleaned column.
print(data.head())


# --- Sentiment Analysis using VADER ---
# VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool.
print("VaderSentiment Score")
# Initialize the VADER sentiment intensity analyzer.
vader_sentiment_score = SentimentIntensityAnalyzer()
# Calculate the compound sentiment score for each cleaned review and add it to a new column.
# The compound score is a single metric that summarizes the sentiment of the text.
data['vader_sentiment_score'] = data['reviewText_Clean'].apply(lambda row: vader_sentiment_score.polarity_scores(row)['compound'])
# Display the head of the dataframe to see the new sentiment score column.
print(data.head())

# --- Categorize Sentiment Scores ---
# Define the bins and corresponding labels for sentiment categorization based on the compound score.
bins = [-1, -0.1 , 0.1, 1]
labels = ['Negative', 'Neutral', 'Positive']
# Create a new column 'vader_sentiment_label' by binning the sentiment scores.
data['vader_sentiment_label'] = pd.cut(data['vader_sentiment_score'], bins=bins, labels=labels)
print("Vader Sentiment Lables")
print(data.head())

print("Sentiment lable bar chart")
data['vader_sentiment_label'].value_counts().plot(kind='bar')
plt.title("VADER Sentiment Analysis Results") # Add a title for clarity
plt.show() # Add this line to display the plot

# --- Sentiment Analysis using Transformers ---
# Specify the model to use to ensure reproducible results and remove the warning.
model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
transformers_pipeline = pipeline("sentiment-analysis", model=model_name)

# Process all reviews in a single batch for much greater efficiency.
# The .tolist() converts the pandas Series to a list.
reviews = data['reviewText_Clean'].tolist()
transformer_results = transformers_pipeline(reviews)

# Extract just the label from each result dictionary and add it to a new column.
data['transformers_sentiment_label'] = [res['label'] for res in transformer_results]

data['transformers_sentiment_label'].value_counts().plot(kind='bar')
plt.title("Transformers Sentiment Analysis Results") # Add a title for clarity
plt.show()
