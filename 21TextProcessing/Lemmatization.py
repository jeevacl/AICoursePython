import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

connect_tokens = [
    # Verb Forms
    'connecting',
    'connected',
    'connects',
    'reconnect',
    'reconnecting',
    'reconnected',
    'disconnect',
    'disconnecting',
    'disconnected',
    'interconnect',
    'interconnecting',
    'misconnect',
    'misconnecting',

    # Noun Forms
    'connection',
    'connections',
    'connectivity',
    'connectedness',
    'connector',
    'disconnection',
    'interconnection',

    # Adjective Forms
    'connectable',
    'connective',
    'unconnected',
    'disconnected',
    'interconnected',
    'hyperconnected',

    # Adverb Forms
    'connectedly',
    'disconnectedly',
    'connectively',
    'drivers',
    'apples'
]
for t in connect_tokens:
    print(f"Root word of - {t:<15} is : {lemmatizer.lemmatize(t)}")
