import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

sent = [
    "this is an example of tokenizer",
    "tokenizer, splits sentences",
    "another example"
]

tokenizer = Tokenizer(lower=True)
tokenizer.fit_on_texts(sent)

print("sentence count -> ", tokenizer.document_count)
print("\n vocab -> ", tokenizer.word_index)
print("\n vocab length -> ", len(tokenizer.word_index))
print("\n matrix shape", tokenizer.texts_to_matrix(sent, mode="binary").shape)

# how does the text look after assigning numbers to the word
sequences = tokenizer.texts_to_sequences(sent)
print("\n showing the sequence\n ",sequences)

vocab_len = len(tokenizer.word_index) + 1
embedding_layer = tf.keras.layers.Embedding(vocab_len, 5) # to represente each with 5 features
result = embedding_layer(tf.convert_to_tensor(sequences)) # always convery to numpy or tensors
print(result.numpy())
result.shape # Shape will be "TensorShape([3, 3, 5])]" (batch_size, input_len, output_dim)


#### output
# sentence count ->  3

# vocab ->  {'example': 1, 'tokenizer': 2, 'this': 3, 'is': 4, 'an': 5, 'of': 6, 'splits': 7, 'sentences': 8, 'another': 9}

# vocab length ->  9

# matrix shape (3, 10)

# showing the sequence
#  [[3, 4, 5, 1, 6, 2], [2, 7, 8], [9, 1]]
