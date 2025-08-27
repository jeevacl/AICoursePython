#!pip install spacy
#!python -m spacy download en_core_web_sm


#import subprocess
#import sys

# Install the spacy library
#subprocess.run([sys.executable, "-m", "pip", "install", "spacy"], check=True)

# Download the English language model
#subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)


import spacy
import pandas as pd

# Load the small English language model from spaCy.
nlp = spacy.load("en_core_web_sm")

# Define the input text
emma_ja = 'emma woodhouse handsome clever and rich with a comfortable home and happy disposition seemed to unite some of the best blessings of existence and had lived nearly twentyone years in the world with very little to distress or vex her she was the youngest of the two daughters of a most affectionate indulgent father and had in consequence of her sisters marriage been mistress of his house from a very early period her mother had died too long ago for her to have more than an indistinct remembrance of her caresses and her place had been supplied by an excellent woman as governess who had fallen little short of a mother in affection sixteen years had miss taylor been in mr woodhouses family less as a governess than a friend very fond of both daughters but particularly of emma between them it was more the intimacy of sisters even before miss taylor had ceased to hold the nominal office of governess the mildness of her temper had hardly allowed her to impose any restraint and the shadow of authority being now long passed away they had been living together as friend and friend very mutually attached and emma doing just what she liked highly esteeming miss taylors judgment but directed chiefly by her own'

# Process the text with the spaCy pipeline.
spacy_doc = nlp(emma_ja)

# --- EFFICIENT IMPLEMENTATION START ---

# 1. Initialize an empty list to hold our data.
token_data = []

# 2. Iterate through each token in the spaCy Doc object.
for token in spacy_doc:
    # 3. Create a dictionary for the current token and append it to the list.
    token_data.append({
        'token': token.text,
        'pos_tag': token.pos_  # Using coarse-grained POS tag
        # 'tag_': token.tag_   # Optional: uncomment to get fine-grained tag
    })

# 4. Create the DataFrame from the list of dictionaries in one single step.
pos_df = pd.DataFrame(token_data)

# --- EFFICIENT IMPLEMENTATION END ---


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

