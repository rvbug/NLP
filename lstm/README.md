# LSTM

## Introduction

Due to the exploding/vanishing gradients in RNNs, LSTM was introduced which can remember longer sequences.

There are 3 gates, forget gate, input gate and output gate.
These gates will help us what to forget, switch to a new context, what to remember and what to pay attention to.
What is the data which it needs to pay attention to


Each gates have their own set of weights which will help them learn (yes they are fully diffentiable)

## Input shape

```python
model.add(LSTM(128, input_shape=(Sequence_length, length_of_unique_words))
# sequence length        - window (see Simple RNN)
# length_of_unique_words - your vocabulary size

```

## Gates

<img width="800" alt="image" src="https://github.com/rvbug/NLP/assets/10928536/87581e09-d93e-456e-b5e5-ddef70e95ce4">


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

Run [this](https://github.com/rvbug/NLP/blob/main/lstm/simple_lstm.py) script

```python


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
