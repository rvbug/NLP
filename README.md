# TOC


`learning path to be added - image`  
`add basics of arrays and matrix from Notion notes`    
`section on input shape of each of the architecture covered here`  
`code section should have all py file or ipynb file or maybe dagshub?`  

# Learning Path

- Arrays, Matrix & Tensors

- Input, Output, Neural Network

- Simple RNN
   * understand input and output shapes
   * simple program

- LSTM
   * understand input and output shapes
   * simple program





# Understanding Arrays, Matrix, Tensors

  ### Create 1D Array
  ```python
    np.array(3)
  ```
  
 
  
  
  ![image](https://user-images.githubusercontent.com/10928536/236743760-0edd86f5-1d7e-4b82-9bac-5a48a35e3b0c.png) 

  <img src="https://your-image-url.type" width="100" height="100">

  ### Create 2D Array
  ```python
  # will create a matrix of 2 rows amd 3 cols
  # you can also use random unform
  # np.random.uniform(size=(2,3))
  np.random.random(size=(2,3)) # or   
  ```
  ![image](https://user-images.githubusercontent.com/10928536/236746538-4482eca2-2ccb-4994-af58-fe3c85ec9a18.png)
  
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
  ![image](https://user-images.githubusercontent.com/10928536/236752424-f2c0e63c-6711-4cf9-bc29-133d3c4d3c0b.png)
  




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





# References
  - [TensorFlow](https://www.tensorflow.org/)
  - [Keras](https://keras.io/api/layers/)
  - [arXiv](https://arxiv.org/)  
  - [Paper with Code](https://paperswithcode.com/)  


# What's next
- Look at my [QNLP Repo](https://github.com/rvbug/QuantumML)  
