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

# EXAMPLE - 3
# Word level




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
