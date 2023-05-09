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

<details>
  <summary><mark><font color=darkred>Simple RNN</font></mark></summary>
  
  ```python
  # for RNN the input shape is `(batch_size, time_step/sequence_length, input_features)
  
  ## np.random.uniform(size=(3, 4, 2)) has same shape as np.random.random([3,4,2])
  
  
  import numpy as np
  # 2 = number of batches
  # 3 = input size
  # 4 = neurons in the o/p layer
  ip = np.random.random([2,3,4])
  ip, ip.shape
  
  wt = np.random.random([4,3])
  wt, wt.shape
  
  op = np.dot(ip, wt)
  op, op.shape
  
  # this is magic
  # 2 batches so we get 4 outputs of 2 batches each
  
  # if we don't give batche number and ip is changed to (3,4)
  # then we get output 
  
  
  # ------------- Using Keras ------------- #
  import tensorflow as tf
  from keras import Sequential
  from keras.layers import Dense, SimpleRNN

  model = Sequential()
  # input shape = 3 timesteps with 4 featues
  # 1 hidden layer with 5 neurons
  model.add(SimpleRNN(5, input_shape=([3,4]), name="input"))
  
  model.add(Dense(1, activation='sigmoid'))
  model.summary()
  tf.keras.utils.plot_model(model, show_shapes=True)
  
  # to check layers' shape and their values
  # i ranges from 0-4 or more depending on your layers
  print(model.get_weights()[i], model.get_weights()[i].shape)
  
   # ------------- Using Keras ------------- #
  
  # you will get output of 2 sets with 4 values each having 3 features -> ([2,4,3])
  
  import tensorflow as tf
  from keras import Sequential
  from keras.layers import Dense, SimpleRNN
  
  simple_rnn = tf.keras.layers.SimpleRNN(10, input_shape=(np.random.random([2, 4,3])))
  # to get all the features and hyperparam use 
  simple_rnn.get_config()
  
 
  ```
  ![image](https://user-images.githubusercontent.com/10928536/236804652-121ef0ce-2323-42fa-92c2-d25c8bbd2000.png)

</details>


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
