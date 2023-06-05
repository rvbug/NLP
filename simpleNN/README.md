# Simple Neural Network


## 1D 
```python

import numpy as np
bh = np.random.uniform(size=(3,4)) # (3,4) 4 = number of neurons and each input has 3 elements
bb = [2,1,4,5] #(4,) - number of hidden neurons in the hidden layer above

ip1 = [1,1,1]
o1 = np.dot(ip1, bh) + bb
print("1 batch i/p and 1 batch o/p ->", o1)
```

## 2D
```python
print("\n")

ip2 = [[1,1,1], [2,2,2]]
o2 = np.dot(ip2, bh) + bb
print("2 batches i/p and 2 batches o/p ->\n", o2)

print("\n")
bip = [[1,2,3], [4,5,6], [7,8,9]] #(3,3)

bo = np.dot(bip, bh) + bb
print("3 batches i/p and 3 batches o/p ->\n", bo)
```

## 3D
```python

import numpy as np

ip3 = np.array([[[1,4,5], [2,4,2]], [[1,1,1], [2,2,2]],[[1,4,9], [2,4,7]],[[5,5,5], [4,4,2]]])
bh = np.random.uniform(size=(3,2))

print(ip3.shape, ip3.shape[0], bh.shape)
o2 = np.dot(ip3, bh)
print(o2)

```
## Neural Network

<img width="820" alt="image" src="https://github.com/rvbug/NLP/assets/10928536/866fce63-3b24-4851-90b9-4174ad2446ba">


## Simple program
[Using Simple Class](https://github.com/rvbug/NLP/blob/main/simpleNN/simple_nn.py)
