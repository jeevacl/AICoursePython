import spacy
from spacy import displacy
from spacy import tokenizer
import re
nlp = spacy.load('en_core_web_sm')
google_text = "Google was founded on September 4, 1998, by computer scientists Larry Page and Sergey Brin while they were PhD students at Stanford University in California. Together they own about 14% of its publicly listed shares and control 56% of its stockholder voting power through super-voting stock. The company went public via an initial public offering (IPO) in 2004. In 2015, Google was reorganized as a wholly owned subsidiary of Alphabet Inc. Google is Alphabet's largest subsidiary and is a holding company for Alphabet's internet properties and interests. Sundar Pichai was appointed CEO of Google on October 24, 2015, replacing Larry Page, who became the CEO of Alphabet. On December 3, 2019, Pichai also became the CEO of Alphabet."
print(google_text)
spacy_doc = nlp(google_text)
for token in spacy_doc.ents:
    print(token.text, token.label_)
displacy.render(spacy_doc, style='ent')

# --- Part 1: Printing entities to the console ---
print("--- Entities found in the text ---")
for entity in spacy_doc.ents:
    # Print the text of the entity and its label
    print(f"Entity: '{entity.text}', Label: '{entity.label_}'")
print("\n" + "=" * 30 + "\n")

google_text_clear = re.sub(r'[^\w\s]', '', google_text).lower()
print('clear google text')
print(google_text_clear)

spacy_doc_clean = nlp(google_text_clear)
print(spacy_doc_clean)
print("\n" + "=" * 30 + "\n")
print("Cleaned google text")
print("\n" + "=" * 30 + "\n")
for token in spacy_doc_clean.ents:
    print(token.text, token.label_)


# --- Part 2: Visualizing the entities in a web browser ---
print("Starting displaCy server to visualize entities...")
print("Open http://127.0.0.1:5000 in your browser.")

# This will start a local web server and block the script until you stop it (Ctrl+C)
docs_to_serve = [spacy_doc, spacy_doc_clean]

displacy.serve(docs_to_serve, style='ent')



if __name__ == "__main__":
    perform_ner()