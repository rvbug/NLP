# Import libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense, SimpleRNN
from keras.utils import pad_sequences
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer


# dataset
# you can use any dataset, but this will help us to understand 
# the basics and how it is fed to the architecture
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

# call the Tokenizer class
# helps in tokenizing the data. For more information, check the readme section
tokenizer = Tokenizer()
# creates unique tokens, assigns the values rathern than writing python code
tokenizer.fit_on_texts(simple_sent)

# shows the count of number of documents = 26
print("sentence count -> ", tokenizer.document_count)

# unique words in the vocabulary = 73
print("\n vocab -> ", tokenizer.word_index)
print("\n vocab length -> ", len(tokenizer.word_index))

# displays the shape of the matrix = (26, 74)
print("\n matrix shape", tokenizer.texts_to_matrix(simple_sent, mode="binary").shape)

# with the numbers assigned to each word, how the sequence will look like
# "The quick brown fox" will be -> [[1, 27, 28, 29], [...].. ]
sequences = tokenizer.texts_to_sequences(simple_sent)
print("\n showing the sequence\n ", sequences)

# the sequences are not in same length so let's pad it to make it same length
sequences = pad_sequences(sequences, padding="pre")
# once padded the shape will be (26, 9) 
# 26 sentences each with pad of 9, I have used 'pre' instead of 'post' padding
# it is a ndarray so it can be directly fed to RNN
print("\n after padding the sequence and shape is \n", sequences.shape, type(sequences) )
print("\n", sequences)

# split the data into x and y
# since we have 9 cols, 1st 8 is for x so we can predict y
x = sequences[:, :8]
y = sequences[:, -1]

# split the data into training and testing 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

print("\n shape is -> ", x_train.shape, x_test.shape, y_train.shape, y_test.shape)
print("\n dimensions are -> ", x_train.ndim, x_test.ndim, y_train.ndim, y_test.ndim)


# creating a sequential model
model = Sequential()

# 1 hidden layer with 5 neuron
# input will be of (Batchsize, input_sequence/time_steps/max_sequence_length, output_features)
# batch size is auto calculated by TF so we need to give the other 2 
# 8 is the max length overall and 1 is the output shape that's why it is (8,1) - check x_train.shape
# if you add another sentence which is greater than 8 words, you need to mention in this i/p shape
# e.g. 
model.add(SimpleRNN(5, input_shape=(8, 1), name="input"))
# output layer is 1 with activation as sigmoid
model.add(Dense(1, activation='sigmoid'))
# displays how much trainable parameters are used (weights and biases)
model.summary()
# tf.keras.utils.plot_model(model, show_shapes=True)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# since this is toy dataset, your accuracy will be bad 
# but you can continue to re-train the model and tune the hyperparameter to make it work
# or better to take a good dataset which can be trained and evaluated 







