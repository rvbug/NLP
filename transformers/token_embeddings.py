import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

sent = [
    "this is an ",
    "tokenizer splits sentences",
    "another example is"
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
# vocab ->  {'is': 1, 'this': 2, 'an': 3, 'tokenizer': 4, 'splits': 5, 'sentences': 6, 'another': 7, 'example': 8}
# vocab length ->  8
# matrix shape (3, 9)
# showing the sequence
#  [[2, 1, 3], [4, 5, 6], [7, 8, 1]]
