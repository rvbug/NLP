# Understanding Arrays, Matrix, Tensors

  ### Create 1D Array
  ```python
    np.array(3)
  ```
 

  <img src="https://user-images.githubusercontent.com/10928536/236743760-0edd86f5-1d7e-4b82-9bac-5a48a35e3b0c.png" width="400" height="200">

  ### Create 2D Array
  ```python
  # will create a matrix of 2 rows amd 3 cols
  # you can also use random unform
  # np.random.uniform(size=(2,3))
  np.random.random(size=(2,3)) # or   
  ```
  <img src="https://user-images.githubusercontent.com/10928536/236746538-4482eca2-2ccb-4994-af58-fe3c85ec9a18.png" width="400" height="200">
  
  
  ### Create 2D Array
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
  <img src="https://user-images.githubusercontent.com/10928536/236752424-f2c0e63c-6711-4cf9-bc29-133d3c4d3c0b.png" width="400" height="200">


  ### Matrix Addition

`Note: this will all make sense during LSTM`
  
  ```python
import numpy as np

a  = np.array([1,2,3])
b = np.array([1,2,3])

print("shape of a ->", a.shape)
print("shape of b ->", b.shape)
print("a matrix ->", a) 
print("b matrix ->", b)
print("a+b = ", a + b)

###### output
# shape of a -> (3,)
# shape of b -> (3,)
# a matrix -> [1 2 3]
# b matrix -> [1 2 3]
# a+b =  [2 4 6]

```
  


 ### Reshape
