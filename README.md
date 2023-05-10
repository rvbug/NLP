# TOC


`learning path to be added - image`  
`add basics of arrays and matrix from Notion notes`    
`section on input shape of each of the architecture covered here`  
`code section should have all py file or ipynb file or maybe dagshub?`  

<details>
  <summary><mark><font color=darkred>Learning Path</font></mark></summary>



</details>


<details>
  <summary><mark><font color=darkred>1D, 2D & 3D Arrays - Memory Representation</font></mark></summary>

  ## Create 1D Array
  ```python
    np.array(3)
  ```
  ![image](https://user-images.githubusercontent.com/10928536/236743760-0edd86f5-1d7e-4b82-9bac-5a48a35e3b0c.png) 

  ## Create 2D Array
  ```python
  # will create a matrix of 2 rows amd 3 cols
  # you can also use random unform
  # np.random.uniform(size=(2,3))
  np.random.random(size=(2,3)) # or   
  ```
  ![image](https://user-images.githubusercontent.com/10928536/236746538-4482eca2-2ccb-4994-af58-fe3c85ec9a18.png)
  
  ## Create 2D Array
  ```python 
  
    import numpy as np
    # shape is (2, 2, 2)
    np.array([
    [[2,3], [4,5]],
    [[6,7], [8,9]]
    ]) 
  
    ## you can also create arrays using
    ## np.random.uniform(size=(3, 4, 2)) which has same shape as np.random.random([3,4,2])
  
  ```
  ![image](https://user-images.githubusercontent.com/10928536/236752424-f2c0e63c-6711-4cf9-bc29-133d3c4d3c0b.png)
  
</details>

<details>
  <summary><mark><font color=darkred>Processing</font></mark></summary>

# Simple Processing
 
  
  ```python
  
  # this processes inputs with one hidden layer of 4 neurons
  # if input is one, we get 1 set (of 4) outputs
  # Batch - if input is two, we get 2 sets (of 4) outputs
  
  class Layers:

  def __init__(self, ip, wt):
    self.ip = ip
    self.wt = np.random.random([(self.ip), wt])
    self.b = np.random.random([wt,])

    print("ip batch is -> ",self.ip)
    print("\n")
    print("wt is", self.wt)
    print("\n")
    print("bias is",self.b)
    print("\n")
  
  def forward(self):
    op = np.dot(self.ip, self.wt) + self.b
    print(op)
  
  ```
  ```python
 
  # single batch of 4 outputs
  l1 = Layers(1, 4)
  l1.forward()
  
  # 3 batch of 4 outputs
  l2 = Layers(3, 4)
  l2.forward()
  
  ```

</details>

# SimpleRNN
  
  ```python

# for RNN the input shape is `(batch_size, time_step/sequence_length, input_features)
# The most difficult part was to understand
# below is the explaination 
  
  
# Very simple sequence of numbers

# EXAMPLE - 1 
a = np.array([10,20, 30,40, 50, 60, 70, 80, 90, 100])

# predict using RNN based on the last 3 i/p sequence
# so create a dataset with 3 i/p features (previous 3) and 1 o/p (next number)
# e.g. first window = [10,20,30,40] => i/p is 10,20,30 and o/p is 40


# steps 
# create a sliding window of 4 (3 prev + 1 next number)
# first window [10,20,30,40]
# where i/p feature = [10,20,30] & o/p feature 40
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

# Finally the input shape will be  -> (7, 3, 1)
# Where 7 will be the sequence number 
  # 7 is total records in the dataset, 1 for each window of len 4
# each record has 3 input feature per record
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
# there is no change in the i/p sequence it will still be (7,3,1) but op will be (7,2) 

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

Sentence - "the quick brown fox jumps over the lazy dog"
We will give two words and predict the 3rd word
i/p - the quick o/p - brown
i/p - quick brown o/p - fox etc

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

```



<details>
  <summary><mark><font color=darkred>Bi-RNN</font></mark></summary>
  

</details>


# References
  - [TensorFlow](https://www.tensorflow.org/)
  - [Keras](https://keras.io/api/layers/)
  - [arXiv](https://arxiv.org/)  
  - [Paper with Code](https://paperswithcode.com/)  


# What's next
- Look at my [QNLP Repo](https://github.com/rvbug/QuantumML)  
