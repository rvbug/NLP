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
