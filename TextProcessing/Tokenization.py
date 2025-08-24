import nltk
nltk.download('punkt_tab')
from nltk.tokenize import  word_tokenize, sent_tokenize

sentence = "Her cat name is Luna. Her dog's name is Max"
sentence_token = sent_tokenize(sentence, "english")
print(sentence_token)


word_token = word_tokenize(sentence,"english",False)
print(word_token)