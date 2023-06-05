# Simple RNN

## RNN Foward Pass

<img width="500" alt="image" src="https://github.com/rvbug/NLP/assets/10928536/9d11eaf0-abfa-4d9a-bcb5-b8f141f868bc">

Sequence information is always mantained -
`o4` is dependent on `x4` and `o3`  
`o3` is dependent on `x3` and `o2` etc..

`Note`: The o in the above diagram is also called hidden state  

`x1` `x2` are all inputs in the form of vectors   
Number of neurons hidden layers will remain the same since o/p is fed back to the same hidden layers


## RNN Backpropogration
<img width="501" alt="image" src="https://github.com/rvbug/NLP/assets/10928536/455b07e9-7175-4bbe-b50a-ab4ac7528368">


## Understanding Input shape


### Input shape:

`input_shape` - These are data (x dependent features & y independent feature)

The input to RNN is in the format 
`(batch_size, time_step or sequence_length, features)`

* `batch_size` is usually kept as None, and have the NN decide (will be in multiples of 8)

* `time_step` is the sequence which you use to train NN or value exists in a sequence  
    e.g [1,2,3,4] = 4 timesteps  

* `feature` how many dimn are used to represente the data in 1 timestep or in another words at each timestep how many features or elements we have  
    e.g. if value is one-hot encoded to 10 values then feature is 10
    
    

```python

# example 
# we have 3 documents each  are 5 in length

import numpy as np

three_five = np.array([
[1,2,3,5,4],
[1,2,3,5,4],
[1,2,3,5,4],
])

# prints (3, 5)
print("original 2D shape", three_five.shape)

# To feed to RNN, we need to covert 2D to 3D by reshaping
reshaped = np.reshape(three, (3,5,1))
print(reshaped)

# [[[1]
#   [2]
#   [3]
#   [5]
#   [4]]
#
#  [[1]
#   [2]
#   [3]
#   [5]
#   [4]]
#
#  [[1]
#   [2]
#   [3]
#   [5]
#   [4]]]

```

Once you have reshaped the data, next would be to feed this to NN one data point at a time.

```python
keras.Input(shape=(5,1))
```


```python

# example 
# if we have 3 documents with 2 input features

import numpy as np
two_five = np.array([
  [
    [1,1,1,1,1],
    [2,2,3,4,5]
  ],
  [
    [1,1,1,1,1],
    [2,2,3,4,5]
  ],
  [
    [1,1,1,1,1],
    [2,2,3,4,5]
  ],  
])


print(two_five)

#5 timesteps and for each timesteps we are collecting 2 features
# array([
# [[1, 1, 1, 1, 1],
# [2, 2, 3, 4, 5]],

# [[1, 1, 1, 1, 1],
# [2, 2, 3, 4, 5]],

# [[1, 1, 1, 1, 1],
# [2, 2, 3, 4, 5]]
])

# we will not be able to feed this directly but has to be reshaped
np.reshape(two_five, (3,5, 2))

# this is because we have 5 sequence/timesteps and each with 2 features


print(two_five.shape) #(3, 2, 5)
# array([[[1, 1],
#         [1, 1],
#         [1, 2],
#         [2, 3],
#         [4, 5]],
#
#        [[1, 1],
#         [1, 1],
#         [1, 2],
#         [2, 3],
#         [4, 5]],
#
#        [[1, 1],
#         [1, 1],
#         [1, 2],
#         [2, 3],
#         [4, 5]]])

```

Input to Keras will be as follows

```python
keras.Input(shape=(5,2))
```


## RNN from scratch
```python
# forward propogation only

import numpy as np

x0 = np.array([[1, 2, 3]])
print("initial inputs", x0.shape)
x1 = np.array([[1, 1, 1]])
print("1st input ", x1.shape)
x2 = np.array([[2, 2, 2]])
print("2nd input ",x2.shape)
w = np.array([[1,2,3], [4,5,6], [5,6,6]])
print("input weight ", w.shape)
w1 = np.array([[1,1,1], [2,2,2], [3,3,3]])
print("output weights ", w1.shape)

### output will be
# initial inputs (1, 3)
# 1st input  (1, 3)
# 2nd input  (1, 3)
# input weight  (3, 3)
# output weights  (3, 3)

o1 = np.dot(x1, w ) + np.dot(x0, w)
print(o1, o1.shape)
o2 = np.dot(x2, w) + np.dot(o1, w1)
print(o2, o2.shape)

### output will be
# [[34 43 48]] (1, 3)
# [[284 290 294]] (1, 3)

```

