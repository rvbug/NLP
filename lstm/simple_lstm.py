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
ht1 = 1  # previous hidden state (o/p)

###### weights ###### 
ht1w1 = 2.70
x1w1 = 1.63
# for tanh
ht1w4 = 1.41
x1w4 = 0.94
# for 2nd sigmoid
ht1w2 = 2.00
x2w2 = 1.65
# for 3 sigmoid
ht1w3 = 4.38
x1w3 = -0.19
###### biases ###### 
bf = 1.62
bti = -0.32
bsi = 0.62 # 2nd sigmoid
bo = 0.59  # output


###### Start calculations ###### 

#1st sigmoid 

y = fn_sigmoid((ht1 * ht1w1) + (xt * x1w1) + bf)
print("first sigmoid output ->")
print_dec(y)
ct1 = ct1 * y
print("value of long term cell state -> ")
print_dec(ct1)


# to tanh  
y_tanh = np.tanh((ht1 * ht1w4) + (x1 * x1w4) + (bti))
print("output of tanh -> ")
print_dec(y_tanh)


#to 2nd Sigmoid  

y_sig2 = fn_sigmoid((ht1 * ht1w2) + (x1 * x2w2) + (bsi))
print("output of 2nd sigmoid -")
print_dec(y_sig2)
ct1 = ct1 + y_sig2 * y_tanh
print("value of long term cell state -> ")
print_dec(ct1)


# final layer  

ct = np.tanh(ct1)
print("cell state last tanh ->")
# this is the pottential short term memory
print_dec(ct)

y_sig3 = fn_sigmoid((ht_1 * ht1w3) + (x1 * x1w3) + (bo))
print("output of 2nd sigmoid ->")
print_dec(y_sig3)
 
## output state 
ht = ct * y_sig3
print("final hidden state ht is  ->")
print("final hidden state becomes ht1")
ht = ht1
print_dec(ht)
print("final cell state of LSTM layer is ->")
print("final cell state becomes ct1")
ct = ct1
print_dec(ct)


################ output ################ 

# first sigmoid output ->
# 1.00
# value of long term cell state -> 
# 1.99
# output of tanh -> 
# 0.97
# output of 2nd sigmoid -
# 0.99
# value of long term cell state -> 
# 2.95
# cell state last tanh ->
# 0.99
# output of 2nd sigmoid ->
# 0.99
# final hidden state ct is  ->
# 0.99
# final cell state of LSTM layer is ->
# 0.99



