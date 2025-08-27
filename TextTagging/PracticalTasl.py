import nltk

from nltk.corpus.reader import titles
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import re
import pandas as pd
import matplotlib.pyplot as plt

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

bbc_data = pd.read_csv("Practice/bbc_news.csv")
print("bbc_data Head Info")
print(bbc_data.head())
print("bbc_data  Info")
print(bbc_data.info())


titles = pd.DataFrame(bbc_data['title'])
print("Titles Head Info")
print(titles.head())


# lowercase
titles['lowercase'] = titles['title'].str.lower()

# stop word removal
en_stopwords = stopwords.words('english')
titles['no_stopwords'] = titles['lowercase'].apply(lambda x: ' '.join([word for word in x.split() if word not in (en_stopwords)]))

# punctuation removal
titles['no_stopwords_no_punct'] = titles.apply(lambda x: re.sub(r'[^\w\s]', '', x['no_stopwords']), axis=1)

# tokenize
titles['tokens_raw'] = titles.apply(lambda x: word_tokenize(x['title']), axis=1)
titles['tokens_clean'] = titles.apply(lambda x: word_tokenize(x['no_stopwords_no_punct']), axis=1)

# lemmatizing
lemmatizer = WordNetLemmatizer()
titles['tokens_clean_lemmatized'] = titles['tokens_clean'].apply(lambda tokens: [lemmatizer.lemmatize(token) for token in tokens])

print("After cleaning data")
print(titles.head())
html_output = titles.head().to_html()

with open('titles_clean.html', 'w', encoding='utf-8') as f:
    f.write(html_output)
    f.close()

print("\nSuccessfully created 'titles_view.html'. You can open this file in a browser.")

# create lists for just our tokens
tokens_raw_list = sum(titles['tokens_raw'], []) #unpack our lists into a single list
tokens_clean_list = sum(titles['tokens_clean_lemmatized'], [])

# Cell 16: Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Cell 17: Process the list of raw tokens
# Assumes 'tokens_raw_list' is a list of strings from a previous step
spacy_doc = nlp(' '.join(tokens_raw_list))
token_data = []

# 2. Iterate through each token in the spaCy Doc object.
for token in spacy_doc:
    # 3. Create a dictionary for the current token and append it to the list.
    token_data.append({
        'token': token.text,
        'pos_tag': token.pos_  # Using coarse-grained POS tag
        # 'tag_': token.tag_   # Optional: uncomment to get fine-grained tag
    })

pos_df = pd.DataFrame(token_data)

# Now the rest of the analysis code remains exactly the same.
print("--- First 15 Tokens and their POS Tags (Efficient Method) ---")
print(pos_df.head(15))
print("\n" + "="*30 + "\n")


# Group by token and POS tag, count occurrences, and sort.
pos_df_counts = pos_df.groupby(['token', 'pos_tag']).size().reset_index(name='counts').sort_values(by='counts', ascending=False)

print("--- Top 20 Most Frequent Token-POS Tag Combinations ---")
print(pos_df_counts.head(20))


pos_df_poscounts = pos_df_counts.groupby(['pos_tag'])['token'].count().sort_values(ascending=False)
print(pos_df_poscounts.head(20))
print("\n" + "="*30 + "\n")

print("noun count")
noun = pos_df_counts[pos_df_counts['pos_tag'] == 'NOUN']
print(noun.head(20))
print("\n" + "="*30 + "\n")


print("adjective count")
adjective = pos_df_counts[pos_df_counts['pos_tag'] == 'ADJ']
print(adjective.head(20))
print("\n" + "="*30 + "\n")

entity_data = []

# 2. Iterate through each entity found in the document.
#    spacy_doc.ents contains the recognized entities.
for entity in spacy_doc.ents:
    # 3. Create a dictionary for the current entity and append it to the list.
    entity_data.append({
        'token': entity.text,
        'ner_tag': entity.label_
    })

# 4. Create the final DataFrame from the list of dictionaries in one single step.
ner_df = pd.DataFrame(entity_data)

# 5. Display the head of the new DataFrame.
print("--- Recognized Entities ---")
print(ner_df.head())

# Cell 26: Group by token and NER tag to get counts, then sort by counts
ner_df_counts = ner_df.groupby(['token', 'ner_tag']).size().reset_index(name='counts').sort_values(by='counts', ascending=False)

# Cell 27: Display the head of the resulting DataFrame (implicitly shows top 5 by default)
ner_df_counts.head()

print("\n" + "="*30 + "\n")
print("Person entity")
people = ner_df_counts[ner_df_counts['ner_tag'] == 'PERSON']
print(people.head())
print("\n" + "="*30 + "\n")