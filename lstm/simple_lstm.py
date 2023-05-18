import numpy as np

# common functions

def print_dec(x):
  print(f"{x:.2f}")

#sigmoid
def fn_sigmoid(x):
  return 1.0/ (1.0 + np.exp(-x))


###### inputs ###### 
ct1 = 2  # previous cell state
xt = 1   # current input
ht1= 1  # previous hidden state (o/p)

###### weights ###### 

wf = 0.1
wi = 0.1
wo = 0.2
wc = 0.3

###### biases ###### 

bf = 0.2
bi = 0.3
bo = 0.1
bc = 0.2

###### forget ###### 
ft = fn_sigmoid((ht1 * wf) + (xt * wf) + bf)
###### input ###### 
it = fn_sigmoid((ht1 * wi) + (xt * wi) + bi)
###### cell  ###### 
ct_hat = np.tanh((ht1 * wc) + (xt * wc) + bc)
ct = ft * ct1 + (it * ct_hat)

###### final state  ###### 
ot = fn_sigmoid((ht1 * wo) + (xt * wo) + bo)
ht = np.tanh(ct) * ot

print("final cell state : ")
print_dec(ct)
print("final hidden state to next layer : ")
print_dec(ht)

############## output ############## 
# final cell state : 
# 1.61
# final hidden state to next layer : 
# 0.57
