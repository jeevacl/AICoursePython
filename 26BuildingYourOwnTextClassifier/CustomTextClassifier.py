# Logistic Regression Model

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Define the dataset as a list of tuples, where each tuple contains a text string and its corresponding sentiment label.
# This dataset will be used to train and test our sentiment classification model.
data = pd.DataFrame([("i love spending time with my friends and family", "positive"),
                     ("that was the best meal i've ever had in my life", "positive"),
                     ("i feel so grateful for everything i have in my life", "positive"),
                     ("i received a promotion at work and i couldn't be happier", "positive"),
                     ("watching a beautiful sunset always fills me with joy", "positive"),
                     ("my partner surprised me with a thoughtful gift and it made my day", "positive"),
                     ("i am so proud of my daughter for graduating with honors", "positive"),
                     ("listening to my favorite music always puts me in a good mood", "positive"),
                     ("i love the feeling of accomplishment after completing a challenging task", "positive"),
                     ("i am excited to go on vacation next week", "positive"),
                     ("i feel so overwhelmed with work and responsibilities", "negative"),
                     ("the traffic during my commute is always so frustrating", "negative"),
                     ("i received a parking ticket and it ruined my day", "negative"),
                     ("i got into an argument with my partner and we're not speaking", "negative"),
                     ("i have a headache and i feel terrible", "negative"),
                     ("i received a rejection letter for the job i really wanted", "negative"),
                     ("my car broke down and it's going to be expensive to fix", "negative"),
                     ("i'm feeling sad because i miss my friends who live far away", "negative"),
                     ("i'm frustrated because i can't seem to make progress on my project", "negative"),
                     ("i'm disappointed because my team lost the game", "negative")
                    ],
                    columns=['text', 'sentiment'])

# Randomly shuffles the rows of the DataFrame.
# frac=1 ensures that all rows are included in the sample.
# reset_index(drop=True) resets the DataFrame index after shuffling, dropping the old index.
data = data.sample(frac=1).reset_index(drop=True)

# Separate features (X) and target (y).
# X will contain the text data, and y will contain the sentiment labels.
X = data['text']
y = data['sentiment']



# --- Vectorize the Text Data ---
# To use text with a machine learning model, we must convert it into a numerical format.
# We'll use the Bag-of-Words (BoW) model, which represents text by the frequency of its words.

# Create an instance of the CountVectorizer.
# This object will convert a collection of text documents to a matrix of token counts.
vectorizer = CountVectorizer()
# This is a crucial two-in-one step:
# 1. fit(): The vectorizer learns the entire vocabulary from our text data (X).
# 2. transform(): It converts each sentence into a numerical vector of word counts.
# The result 'countvec_fit' is a sparse matrix, an efficient format for data with many zeros.
countvec_fit = vectorizer.fit_transform(X)
# Convert the sparse matrix to a dense array and then to a Pandas DataFrame.
# The column names are the words (features) learned by the vectorizer.
bag_of_words = pd.DataFrame(countvec_fit.toarray(), columns=vectorizer.get_feature_names_out())


# --- Split Data into Training and Testing Sets ---
# This is a critical step in machine learning to evaluate the model's performance on unseen data.
# We split our features (bag_of_words) and our labels (y) into two corresponding sets.
# X_train, X_test: Features for training and testing.
# y_train, y_test: Labels for training and testing.
X_train, X_test, y_train, y_test = train_test_split(
    bag_of_words,  # The feature data (our word counts)
    y,             # The target labels (the sentiment)
    test_size=0.3, # Reserve 30% of the data for the test set (and 70% for training).
    random_state=7 # Ensures the split is the same every time we run the code, for reproducibility.
)
# Print the shapes of the resulting datasets to verify the split.
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")

# --- Train and Evaluate Logistic Regression Model ---

# Initialize the Logistic Regression model.
# random_state ensures reproducibility of the model's training process.
lr = LogisticRegression(random_state=1).fit(X_train, y_train)

# Make predictions on the test set using the trained Logistic Regression model.
y_pred_lr = lr.predict(X_test)

# Calculate the accuracy of the model by comparing predicted labels (y_pred_lr) with actual labels (y_test).
accuracy = accuracy_score(y_pred_lr, y_test)
print(f"\nAccuracy of Logistic Regression Model: {accuracy:.2f}")

# Generate a detailed classification report.
# This report includes precision, recall, f1-score, and support for each class.
# zero_division=0 handles cases where a class has no predicted samples, preventing errors.
print(classification_report(y_test, y_pred_lr, zero_division=0))