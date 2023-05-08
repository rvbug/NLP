# TOC


`learning path to be added - image`  
`add basics of arrays and matrix from Notion notes`    
`section on input shape of each of the architecture covered here`  

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
  ```
  ![image](https://user-images.githubusercontent.com/10928536/236752424-f2c0e63c-6711-4cf9-bc29-133d3c4d3c0b.png)
  
</details>

<details>
  <summary><mark><font color=darkred>Processing</font></mark></summary>

# Simple Processing
  ```python
  
  import numpy as np
  class Layer:

  def __init__(self, ip_size, n_neurons):
    self.w = np.random.uniform(size=(n_neurons, ip_size)).T
    self.b = np.random.rand(n_neurons)
  
  def forward_pass(self, ip):
    self.ip = ip
    print("\n--------\n")
    print("\n weight is \n", self.w)
    print("\n bias is", self.b)
    print("\n bias is", self.ip)
    self.output = np.dot(self.ip, self.w) + self.b 
    
  
  # input size can keep varying 
  input_size = 3
  input = np.random.rand(input_size)

  # first Layer
  L1 = Layer(input_size, 4)
  L1.forward_pass(input)
  print("\n output is \n", L1.output)

  # second layer which has 
  # 1st layer's output as input &
  # 3 neuron in the second layer

  L2 = Layer(L1.output.shape[0], 3)
  L2.forward_pass(L1.output)
  print("\n output is \n", L2.output)

  L3 = Layer(L2.output.shape[0], 2)
  L3.forward_pass(L2.output)
  print("\n output is \n", L3.output)
  
```


# Batch Processing
  
  ```python
  import numpy as np
  
  bh = np.random.uniform(size=(3,4)) # (3,4) 4 = number of neurons and each input has 3 elements
  bb = [2,1,4,5] #(4,) - number of hidden neurons in the hidden layer above

  ip1 = [1,1,1]
  o1 = np.dot(ip1, bh) + bb
  print("1 batch i/p and 1 batch o/p ->", o1)

  print("\n")

  ip2 = [[1,1,1], [2,2,2]]
  o2 = np.dot(ip2, bh) + bb
  print("2 batches i/p and 2 batches o/p ->\n", o2)

  print("\n")
  bip = [[1,2,3], [4,5,6], [7,8,9]] #(3,3)

  bo = np.dot(bip, bh) + bb
  print("3 batches i/p and 3 batches o/p ->\n", bo)

  ```

</details>

<details>
  <summary><mark><font color=darkred>Code</font></mark></summary>
  
# Code
- Simple neuron
- Simple Neuron using Python Class
- Batch Simple Neuron
- Batch Neuron using Python Class

</details>


- Image from iPad


# References

  - [arXiv](https://arxiv.org/)  
  - [Paper with Code](https://paperswithcode.com/)  


# What's next
- Look at my [QNLP Repo](https://github.com/rvbug/QuantumML)  
