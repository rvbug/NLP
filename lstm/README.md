# LSTM

## Introduction

Due to the exploding/vanishing gradients in RNNs, LSTM was introduced which can remember longer sequences.

There are 3 gates, forget gate, input gate and output gate.
These gates will help us what to forget, switch to a new context, what to remember and what to pay attention to.
What is the data which it needs to pay attention to


Each gates have their own set of weights which will help them learn (yes they are fully diffentiable)

## Gates

<img width="500" alt="image" src="https://github.com/rvbug/NLP/assets/10928536/912354ef-5414-4945-81e7-cee09b18cd96">


## Simple code

### Description
```python

### biases

# bf = bias forget gate
# bti = bias tanh i/p gate
# bsi = bias sigmoid i/p gate
# bio = boas o/p gate


### inputs
# ht1 = previous hidden state
# ct1 = previous cell state
# xt = inputs

### weights

# ht1w1 = wt to 1st sigmoid
# ht1w2 = wt to 1st sigmoid
# ht1w3 = wt to 1st sigmoid
# ht1w4 = wt to 1st sigmoid
# x1w1 = wt i/p to 1st sigmoid
# x2w2 = wt on st tanh 
# x3w3 = wt i/p to 2nd sigmoid
# x4w4 = wt i/p to final sigmoid

### final outputs
# ht = current hidden state
# ct = current cell state


```

### Code

```python
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
final hidden state ct is  ->
print_dec(ht)
print("final cell state of LSTM layer is ->")
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

```














# References
[2019 LSTM Paper](https://arxiv.org/pdf/1909.09586.pdf)  
[Chris Olah LSTM Introduction](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)  
[YT - StatQuest](https://youtu.be/YCzL96nL7j0)  
[YT - MIT Lecture](https://youtu.be/ySEx_Bqxvvo)  
[YT - Krish](https://www.youtube.com/watch?v=FLjn0H2bCvA)  
