import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical
from keras import Sequential
from keras.layers import Dense, SimpleRNN

movies = pd.read_csv('/content/tmdb_5000_movies.csv')

df_movies = movies['original_title']
print(type(df_movies))
df_movies = df_movies.to_list()
print(type(df_movies))
print(len(df_movies))

# tokenizer = Tokenizer(lower=True)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_movies)
# how does the text look after assigning numbers to the word
sequences = tokenizer.texts_to_sequences(df_movies)
sequences[:3]

# print("sentence count -> ", tokenizer.document_count)
# print("\n vocab -> ", tokenizer.word_index)
# print("\n vocab length -> ", len(tokenizer.word_index))
# print("\n matrix shape", tokenizer.texts_to_matrix(df_movies, mode="binary").shape)
# print("\n showing the sequence\n ", sequences)
# to view the first 3 items in the dictionary
# dict(list(tokenizer.word_index.items())[0: 3])
len(tokenizer.word_index), type(tokenizer.word_index)

x = []
y = []
for i in sequences:
  # pick the ones which has more than 1 word 
  if len(i) > 1:
    for idx in range(1, len(i)):
      x.append(i[:idx])
      y.append(i[idx])

print(sequences[:8], len(sequences))
print(x[:1], len(x))
print(y[1], len(y))


# pad x so we have same sequence length
# largest sentence is 14 words long
# we have 8483 sentences or movie names in this case
x = pad_sequences(x)
print(x.shape, type(x))

# convert vectors to binary class for y
y = to_categorical(y)
print(y.shape, type(y))

# why 5045 since the total unique words are 5043 but 
# we add +1 for padding 

# our vocabulary is always +1 and will be
# unique words in our dictionary
vocab = len(tokenizer.word_index) + 1
print(vocab)

# Embedding(ip_dimn, output_dimn, max_ip_len)
# SimpleRNN(hidden_layers in a neuron)
# Dense(vocab_size, activation='softmax')


model = Sequential()
model.add(tf.keras.layers.Embedding(input_dim=5045, output_dim=10, input_length=14))
model.add(SimpleRNN(units=64))
model.add(Dense(units=vocab, activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x, y, epochs=100, validation_data=(x, y), verbose=0)


#--------------prediction------------------#
def make_prediction(text, n_words):
    for i in range(n_words):
        text_tokenize = tokenizer.texts_to_sequences([text])
        text_padded = tf.keras.preprocessing.sequence.pad_sequences(text_tokenize, maxlen=14)
        prediction = np.squeeze(np.argmax(model.predict(text_padded), axis=-1))
        prediction = str(vocab_array[prediction - 1])
        print(vocab_array[np.argsort(model.predict(text_padded)) - 1].ravel()[:-3])
        text += " " + prediction
    return text
  
  
  
make_prediction("Dark", 3)
make_prediction("Kill", 3)
make_prediction("The man", 5)


