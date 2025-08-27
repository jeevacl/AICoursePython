sentence_1 = "I had a great time at the movie it was really funny"
sentence_2 = "I had a great time at the movie but the parking was terrible"
sentence_3 = "I had a great time at the movie but the parking wasn't great"
sentence_4 = "I went to see a movie"

# From the "TextBlob" section
from textblob import TextBlob
print("\n" + "="*30 + "\n")

# --- Analysis for Sentence 1 ---
print("Sentence 1: " + sentence_1)
sentence_1_score = TextBlob(sentence_1).sentiment.polarity
# FIXED: Converted the score (a number) to a string before printing
print("Score : " + str(sentence_1_score))
print("\n" + "="*30 + "\n")


# --- Analysis for Sentence 2 ---
# FIXED: The redundant str() around sentence_2 is removed
print("Sentence 2: " + sentence_2)
sentence_2_score = TextBlob(sentence_2).sentiment.polarity
print("Score : " + str(sentence_2_score))
print("\n" + "="*30 + "\n")


# --- Analysis for Sentence 3 ---
print("Sentence 3: " + sentence_3)
sentence_3_score = TextBlob(sentence_3).sentiment.polarity
print("Score : " + str(sentence_3_score))
print("\n" + "="*30 + "\n")


# --- Analysis for Sentence 4 ---
print("Sentence 4: " + sentence_4)
sentence_4_score = TextBlob(sentence_4).sentiment.polarity
print("Score : " + str(sentence_4_score))
print("\n" + "="*30 + "\n")


# From the "VADER" section
# This import is correct and ready for the next part of your course
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# The analyzer object is now named 'vader_sentiment'
vader_sentiment = SentimentIntensityAnalyzer()

print("\n" + "="*30 + "\n")
print("--- VADER Sentiment Analysis ---")
print("\n" + "="*30 + "\n")


# --- Analysis for Sentence 1 ---
print("Sentence 1: " + sentence_1)
# Using the new variable name to get the scores
sentence_1_scores = vader_sentiment.polarity_scores(sentence_1)
sentence_1_compound_score = sentence_1_scores['compound']
print("VADER Score (Compound): " + str(sentence_1_compound_score))
print("\n" + "="*30 + "\n")


# --- Analysis for Sentence 2 ---
print("Sentence 2: " + sentence_2)
sentence_2_scores = vader_sentiment.polarity_scores(sentence_2)
sentence_2_compound_score = sentence_2_scores['compound']
print("VADER Score (Compound): " + str(sentence_2_compound_score))
print("\n" + "="*30 + "\n")


# --- Analysis for Sentence 3 ---
print("Sentence 3: " + sentence_3)
sentence_3_scores = vader_sentiment.polarity_scores(sentence_3)
sentence_3_compound_score = sentence_3_scores['compound']
print("VADER Score (Compound): " + str(sentence_3_compound_score))
print("\n" + "="*30 + "\n")


# --- Analysis for Sentence 4 ---
print("Sentence 4: " + sentence_4)
sentence_4_scores = vader_sentiment.polarity_scores(sentence_4)
sentence_4_compound_score = sentence_4_scores['compound']
print("VADER Score (Compound): " + str(sentence_4_compound_score))
print("\n" + "="*30 + "\n")


from tabulate import tabulate  # The library for creating tables
# --- 2. Data Setup ---
# The list has been renamed to 'input_texts' for easier integration
input_texts = [
    "I had a great time at the movie it was really funny",
    "I had a great time at the movie but the parking was terrible",
    "I had a great time at the movie but the parking wasn't great",
    "I went to see a movie"
]

# --- 3. Analyzer Initialization ---
# Create the VADER analyzer object once
vader_sentiment1 = SentimentIntensityAnalyzer()

# --- 4. Data Processing ---
# Create a list to hold all our results
results_data = []

# Loop through each sentence in the renamed list to analyze it
for sentence in input_texts:
    # --- TextBlob Analysis ---
    tb_score = TextBlob(sentence).sentiment.polarity

    # --- VADER Analysis ---
    vader_scores = vader_sentiment1.polarity_scores(sentence)
    vader_compound_score = vader_scores['compound']

    # --- Store the results for this sentence ---
    # Append a list containing the sentence and its scores
    results_data.append([sentence, tb_score, vader_compound_score])


# --- 5. Display Results in a Table ---
# Define the headers for our table
headers = ["Sentence Text", "TextBlob Polarity", "VADER Compound"]

# Use the tabulate function to create and print the table
# The 'grid' format looks great in the console
print("\n--- Sentiment Analysis Comparison ---")
print(tabulate(results_data, headers=headers, tablefmt="grid", floatfmt=".4f"))