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


# Imagine you are teaching a student (the machine learning model) to identify spam emails.
# The Components: X and y
# First, let's understand the two basic parts of your dataset:
# X (Features): This is the data used to make a prediction. It's the evidence. In our analogy, X would be the content of the emails (the words, the sender, etc.). These are the "questions" you give the student.
# y (Labels): This is the correct answer or the thing you want to predict. In our analogy, y would be the label for each email, either "spam" or "not spam." These are the "answers" to the questions.

# You start with a large dataset containing both the emails (X) and their correct labels (y).
# The Split: Training vs. Testing
# Now, you need to both teach the student and give them a final exam to see if they actually learned. You can't use the same material for both studying and the exam, because then the student might just memorize the answers without learning the concepts.
# This is why you split your dataset into two parts: a training set and a testing set.
# The Four Resulting Pieces
# 1. X_train (Training Features)
# What it is: A large portion (usually 70-80%) of your email content (X).
# Analogy: These are the practice questions and study materials you give to the student. The student is allowed to look at these emails and analyze them as much as they want to learn the patterns of what makes an email spam.
# 2. y_train (Training Labels)
# What it is: The corresponding correct "spam" or "not spam" labels for every email in X_train.
# Analogy: These are the answers to the practice questions. While studying, the student looks at an email (X_train) and makes a guess. Then they look at the correct answer (y_train) to see if they were right. This process of checking against the answers is how the student (the model) learns.
# 3. X_test (Testing Features)
# What it is: The remaining smaller portion (usually 20-30%) of your email content (X). This data was kept separate and was never seen during training.
# Analogy: This is the final exam paper. It contains questions the student has never seen before. You hand this to the student after they have finished studying.
# 4. y_test (Testing Labels)
# What it is: The corresponding correct "spam" or "not spam" labels for the emails in X_test.
# Analogy: This is the teacher's secret answer key for the final exam. The student does not get to see this. After the student has made their predictions on the exam questions (X_test), you, the teacher, will use this answer key (y_test) to grade their performance.
# Summary Table
# Variable	Role	Analogy	Purpose
# X_train	Training Features	Practice Questions	To give the model data to learn patterns from.
# y_train	Training Labels	Answers to Practice Questions	To allow the model to check its understanding and adjust its logic. This is the learning part.
# X_test	Testing Features	The Final Exam Paper	To present the model with new, unseen data.
# y_test	Testing Labels	The Teacher's Answer Key	To evaluate how well the model performs by comparing its predictions against the true answers.
# Why is this separation so critical?
# The goal of a machine learning model is to generalize—to be good at making predictions on new data it has never encountered before.
# By separating your data into training and testing sets, you can simulate this real-world scenario. The model's performance on the test set (X_test, y_test) gives you a fair and unbiased estimate of how it will perform when you deploy it to handle brand-new data
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

# --- Train and Evaluate Naive Bayes Model ---
# Multinomial Naive Bayes is a probabilistic algorithm that is particularly well-suited for
# text classification tasks involving word counts.

# Import the Multinomial Naive Bayes classifier from scikit-learn.
from sklearn.naive_bayes import MultinomialNB

# Initialize the Multinomial Naive Bayes model and train it in one step.
# .fit(X_train, y_train) teaches the model by calculating the probability of each word
# appearing in each class (positive vs. negative) based on the training data.
naive_bayes = MultinomialNB().fit(X_train, y_train)

# Use the trained Naive Bayes model to make predictions on the unseen test features (X_test).
y_pred_nb = naive_bayes.predict(X_test)

# Calculate and print the accuracy of the Naive Bayes model.
print("\nAccuracy of Naive Bays Model")
# Sample Output: 0.83
print(accuracy_score(y_test, y_pred_nb))

# Generate and print a detailed classification report for the Naive Bayes model.
print("classification_report for Naive Bays Model")
# Sample Output:
#               precision    recall  f1-score   support
#
#     negative       0.75      1.00      0.86         3
#     positive       1.00      0.67      0.80         3
#
#     accuracy                           0.83         6
#    macro avg       0.88      0.83      0.83         6
# weighted avg       0.88      0.83      0.83         6
print(classification_report(y_test, y_pred_nb, zero_division=0))

# --- Analysis of Classification Report Metrics ---

# --- Core Concepts Explained of  Naive Bays Model Output ---
#
# Precision: Of all the times the model predicted a certain class (e.g., "positive"), how often was it actually correct?
#            This metric is about the quality of the predictions.
#            High Precision: When the model predicts a class, it's very likely to be right.
#
# Recall (Sensitivity): Of all the actual instances of a class, how many did the model successfully find?
#                       This metric is about the completeness of the predictions.
#                       High Recall: The model is good at finding all instances of a class.
#
# F1-Score: The harmonic mean (a type of average) of Precision and Recall. It provides a single score that balances both metrics.
#           It's especially useful when you have an uneven class distribution (imbalanced dataset).
#
# Support: This is simply the number of actual occurrences of the class in your dataset. It gives context to the other metrics.
#
# --- Analysis of Your Model's Performance ---
# Here’s what a specific output might be telling you:
#
# === negative Class ===
# Precision: 0.00
# This means that every single time the model predicted an instance was "negative," it was wrong.
#
# Recall: 0.00
# This means the model failed to identify any of the actual "negative" instances that existed in the data.
#
# F1-Score: 0.00
# Since both precision and recall are zero, the F1-score is also zero.
#
# Support: 1
# This is the key to understanding the problem. There was only one actual "negative" sample in the test data.
#
# Conclusion for negative: The model is completely failing on this class. It has not learned how to identify "negative" instances at all.
#
# === positive Class ===
# Precision: 0.80 (80%)
# When the model predicted an instance was "positive," it was correct 80% of the time.
#
# Recall: 0.80 (80%)
# The model successfully found 80% of all the actual "positive" instances in the data.
#
# F1-Score: 0.80 (80%)
# This is a good, balanced score for this class.
#
# Support: 5
# There were five actual "positive" samples in the test data.
#
# Conclusion for positive: The model is reasonably effective at identifying "positive" instances.
#
# --- Overall Averages and Accuracy ---
#
# Accuracy: 0.67 (67%)
# This is the overall percentage of correct predictions ((4 correct positive + 0 correct negative) / 6 total samples = 66.7%).
# This number can be very misleading! It makes the model seem okay, but it completely hides the fact that it's failing on one of the classes.
#
# Macro Avg: 0.40 (40%)
# This is the simple average of the precision, recall, and F1-scores across both classes, treating each class equally.
# This low score accurately reflects that the model is performing poorly overall because it fails on one class.
#
# Weighted Avg: 0.67 (67%)
# This is the average weighted by the support of each class. Since the "positive" class has 5 samples and "negative" only has 1,
# this average is heavily skewed towards the performance on the "positive" class.
#
# --- The Main Takeaway ---
# A small, imbalanced dataset can cause a model to learn a lazy strategy: it becomes biased towards predicting the majority class.
# It may perform reasonably well on the majority class but completely fail to learn the characteristics of the rare, minority class.
# In a real-world scenario (like medical diagnosis or fraud detection), this would be a very dangerous model,
# as it would miss every instance of the rare but critical event.


#------------------------------------------------------------------#
#Linear Support Vector Machine
#------------------------------------------------------------------#
# using LinearSVC algorithm
from sklearn.svm import LinearSVC
svm = LinearSVC(random_state=1).fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("\nAccuracy of SVM Model")
print(accuracy_score(y_test, y_pred_svm))
print("classification_report for SVM Model")
print(classification_report(y_test, y_pred_svm, zero_division=0))

# using SGDClassifier algorithm
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(random_state=1).fit(X_train, y_train)
y_pred_sgd = sgd.predict(X_test)
print("\nAccuracy of SGD Model")
print(accuracy_score(y_test, y_pred_sgd))
print("classification_report for SGD Model")
print(classification_report(y_test, y_pred_sgd, zero_division=0))
