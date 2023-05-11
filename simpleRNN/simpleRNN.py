# libraries

import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, SimpleRNN
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences


# dataset
simple_sent = [
    "The quick brown fox",
    "jumps over a lazy dog",
    "the lazy dog is sleeping",
    "I like to work on DL",
    "DL is amazing",
    "DL is awesome to learn"
    "DL the future",
    "I like pytorch better",
    "tensorflow is meh",
    "i am trying to find new sentences",
    "to help validate the dataset",
    "checking what else can I write",
    "I need to make NN to learn",
    "need to split the dataset",
    "thats why I am writing such a long sentences",
    "the cat sat on the mat which is good",
    "the dog ate my homework",
    "hello world",
    "this is a test",
    "python is awesome",
    "the cat sat on the mat",
    "the dog ate my homework",
    "I like pizza",
    "she is a good singer",
    "what else can I write to generate",
    "maybe lorem ipsum",
    "who knows, I am ending it here"
]

# tokenize the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(simple_sent)

# check the output
print("sentence count -> ", tokenizer.document_count)
print("\n vocab -> ", tokenizer.word_index)
print("\n vocab length -> ", len(tokenizer.word_index))
print("\n matrix shape", tokenizer.texts_to_matrix(simple_sent, mode="binary").shape)

# how does the text look after assigning numbers to the word
sequences = tokenizer.texts_to_sequences(simple_sent)
print("\n showing the sequence\n ", sequences)

# the sequences are not in same length so let's pad it
sequences = pad_sequences(sequences, padding="pre")
print("\n after padding the sequence and shape is \n", sequences.shape, type(sequences) )

print("\n", sequences)

# split the data into x and y
x = sequences[:, :8]
y = sequences[:, -1]

# split the data into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

print("\n shape is -> ", x_train.shape, x_test.shape, y_train.shape, y_test.shape)
print("\n dimensions are -> ", x_train.ndim, x_test.ndim, y_train.ndim, y_test.ndim)


model = Sequential()

model.add(SimpleRNN(5, input_shape=(8, 1), name="input"))

model.add(Dense(1, activation='sigmoid'))
model.summary()
# tf.keras.utils.plot_model(model, show_shapes=True)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
