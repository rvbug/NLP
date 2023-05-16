# Simple RNN

## RNN structure

<img width="200" alt="image" src="https://github.com/rvbug/NLP/assets/10928536/0d975713-3c65-4718-9e7b-2c3ce011eeba">


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


