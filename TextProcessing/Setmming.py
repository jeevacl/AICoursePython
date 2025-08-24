from nltk.stem import PorterStemmer
ps = PorterStemmer()
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
    'connectively'
]
for t in connect_tokens:
    print(f"Root word of - {t:<15} is : {ps.stem(t)}")
for t in connect_tokens:
    print(f"Root word of - {t:>15} is : {ps.stem(t)}")
