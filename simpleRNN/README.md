# Simple RNN

```python

# input shape is `(batch_size, time_step or sequence_length, features)
- batch_size is None, and we can have the NN decide it
- time_step is the sequence which you use to train NN or value exists in a sequence  
e.g [1,2,3,4] = 4 timesteps  
- feature how many dimn are used to represente the data in 1 timestep
e.g. if value is one-hot encoded to 10 values then feature is 10


# Very simple sequence of numbers

# EXAMPLE - 1 
a = np.array([10,20, 30,40, 50, 60, 70, 80, 90, 100])

# predict using RNN based on the last 3 i/p sequence
# so create a dataset with 3 i/p features (previous 3) and 1 o/p (next number)
# e.g. first window = [10,20,30,40] => i/p is 10,20,30 and o/p is 40


# steps 
# create a sliding window of 4 
# first window [10,20,30,40]
# second window [20,30,40,50]
# slide across for the entire dataset
# Data will finally looks like this

# input features        # output features
# [10,20,30,40]               50
# [20,30,40,50]               60
# [30,40,50,60]               70
# [40,50,60,70]               80
# [50,60,70,80]               90
# [60,70,80,90]               100

# Finally the input shape will be  -> (6, 4, 1)
# Where 6 will be the sequence number 
# 6 is total records in the dataset, 1 for each window of len 4
# each record has 4 input feature per record
# each input has single number which is 1

# EXAMPLE - 2 

np.array([
  [[1,2], [2,3], [3,4]],
  [[2,3], [3,4], [4,5]],
  [[3,4], [4,5], [5,6]],
  [[4,5], [5,6], [6,7]],
  [[5,6],[6,7], [7,8]],
  [[6,7], [7,8], [8,9]],
  [[7,8], [8,9], [9,10]]

])

# i/p to RNN (7,3,2) where there we have same 7 records
# each with 3 input sequence and evey sequence having 2 values
# output will be (7,1)

# If I have to pedict 2 numbers instead of 1 number
# there is no change in the i/p sequence it will still be (7,3,2) but op will be (7,2) 

# For Word

# input shape will be (num_sentences, max_sequence_length ,embedding_size)
# num_sentences       - number of sentences in the dataset
# max_sequence_length - is the max length in whole of the dataset
# embedding_size      - size of word being represented as embedding  

# output shape will be (num_sentences, embedding_size)
# where each element in the o/p will be probability of word in the vocab
# num_sentences  - number of words which needs to be predicted
# embedding_size - representation of the word

# EXAMPLE 

#Sentence - "the quick brown fox jumps over the lazy dog"
#We will give two words and predict the 3rd word
#i/p - the quick o/p - brown
#i/p - quick brown o/p - fox etc

step 1 - create a dictionary for individual words
[0,1,2,3,4,5,6,7,0,8]

# input           # output 
# [0,1]               2
# [1,2]               3
# [2,3]               4
# [3,4]               5
# [4,5]               6
# [5,0]               0
# [0,6]               7

# shape is (7,2,1)
# 7 = input sequence
# 2 = window size
# 1 = 1 feature for each word

X = [
[[0],[1]], [[1], [2]], [[2],[3]],[[3],[4]],[[4],[5]],[[5],[0]],[[0],[6]]
]

# shape is (7,1)
# because we are predicting 1 value for all 7 sequences of 2 words each

y = [
[2],[3],[4],[5],[6],[0],[7]
]


# EXAMPLE (using one-hot)

sent1 = The quick brown fox
sent2 = jumps over the lazy dog
sent3 = she sells seashells by the seashore
sten4 = Peter piper picked a peak of pickled piper

# tokenized
# there will be 22 unique words on the volab
# now represent each word and pad to the largest sentence (8)
sent1 = [1,2,3,4,0,0,0,0]
sent2 = [5,6,7,8,9,0,0,0]
sent3 = [10,11,12,13,14,0,0,0]
sent4 = [15,16,17,18,19,20,21,22]

# input shape (4,8,1)
# 4 sentences which each sent having len of 8
# 1 is output feature

# output shape (4,1)
# Since there will be 4 outputs with 1 feature

# EXAMPLE (using 2 features)



# EXAMPLE (using word embeddings)


# Typical step would be as follows for the example
lst = ["I like to eat bananas.", "Bananas are yellow."]

# Tokenize the text
lst = ["I", "like", "to" , "eat" , "bananas", "." , "Bananas" "are" "yellow", "."]
# assign unique index to each of them
dt = {"I" : 0 , "like" : 1 , "to" : 2 , "eat" :3 , "bananas": 4, ".": 5 , "Bananas": 6 , "are": 7,  "yellow" :8 }
# convert the text to integers
t2i = [[0,1,2,3,4,5],[6,7,8, 5]]
# pad to make it equal length of 10 each
t2i = [[0,1,2,3,4,5,0,0,0,0],[6,7,8,5,0,0,0,0,0,0]]
# create word embedding (e.g. Word2Vec or GloVe)
# shape (9,3) - 9 is the unique words in vocab and 3 is the word representation of each word (3 dim vec)
[[0.1, 0.2, 0.3],
[0.4, 0.5, 0.6],
[0.4, 0.3, -0.6],
[0.3, 0.5, 0.6],
[0.4, 0.5, 0.3],
[0.4, -0.5, 0.2],
[0.7, 0.5, -0.6],
[0.8, 0.6, 0.3],
[-0.1, 0.5, 0.6]]
# create input output pairs
# input                 # output 
# [0,1,2,3,4]               5
# [1,2,3,4,5]               0
# [2,3,4,5,6]               7
# [3,4,5,6,7]               8
# convert i/p to word embeddings
[[
[0.1, 0.2, 0.3],
[0.4, 0.5, 0.6],
[0.4, 0.3, -0.6],
[0.3, 0.5, 0.6],
[0.4, 0.5, 0.3]]
[[
[0.8, 0.6, 0.3],
[-0.1, 0.5, 0.6]]
]]

# final steps for RNN is as follows

```
