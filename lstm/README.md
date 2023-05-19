# LSTM

## Introduction

Due to the exploding/vanishing gradients in RNNs, LSTM was introduced which can remember longer sequences.

There are 3 gates, forget gate, input gate and output gate.
These gates will help us what to forget, switch to a new context, what to remember and what to pay attention to.
What is the data which it needs to pay attention to

Each gates have their own set of weights , this makes them [differentiable](https://en.wikipedia.org/wiki/Differentiable_function) helping them to learn

To understand LSTM, the following concepts must be clear

## Concat
```python
x = np.random.random((5,1))
y = np.random.random((6,1))

print(x,"\n\n", y)
z = np.concatenate((x, y))
print("\n" z)
print(z.shape)

########## output

# [[0.61453294]
#  [0.50259992]
#  [0.01336285]
#  [0.26558368]
#  [0.29504331]] 

#  [[0.78379103]
#  [0.52628384]
#  [0.97248502]
#  [0.0859092 ]
#  [0.23836397]
#  [0.35459217]]

# array([[0.61453294],
#        [0.50259992],
#        [0.01336285],
#        [0.26558368],
#        [0.29504331],
#        [0.78379103],
#        [0.52628384],
#        [0.97248502],
#        [0.0859092 ],
#        [0.23836397],
#        [0.35459217]])

# (11,1)

```

## Addition 
```python
s = np.random.random((4,1))
t = np.random.random((4,))
print(s.shape, t.shape )
u = s + t
print("after adding -> ", u.shape)

##### output 
# (4, 1) (4,)
# after adding ->  (4, 4)
```

## Hardamard 
```python
a = [1,2,3]
b = [3,2,1]

np.multiply(a, b)

### output
[3,4,3]

```

## Gates

<img width="771" alt="image" src="https://github.com/rvbug/NLP/assets/10928536/7853ea86-5882-4f57-a061-e12c3c562400">


## Simple code

Find optimized code [here](https://github.com/rvbug/NLP/blob/main/lstm/lstm.py)


## Input shape

```python
model.add(LSTM(128, input_shape=(Sequence_length, length_of_unique_words))
# sequence length        - window (see Simple RNN)
# length_of_unique_words - your vocabulary size

```


# References
[2019 LSTM Paper](https://arxiv.org/pdf/1909.09586.pdf)  
[Chris Olah LSTM Introduction](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)  
[YT - StatQuest](https://youtu.be/YCzL96nL7j0)  
[YT - MIT Lecture](https://youtu.be/ySEx_Bqxvvo)  
[YT - Krish](https://www.youtube.com/watch?v=FLjn0H2bCvA)  
