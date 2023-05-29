# Introduction

Before understanding what Transformers are, it is important to undersatand what `Attention` & `Self Attention` mechanisms are. 
This concept made the Transformers one of the biggest breakthroughs in Deep Learning.

As humans, we tend to concentrate on the things that matters the most i.e. we pay more attention to the things which are interesting. The same concept can be applied to machines which is what this section will take about.

Attention were first used  `Encoder-Decoder` Architecture where the final output of RNN/LSTM creates a **Context Vector**. 
If setences were longer, these vectors were still unable to capture semantic meaning. 
RNNs also had problems dealing with large sentences and on the ther hand, LSTMs were slow and process inputs sequentially. <br>

So there has to be a way where these context vectors should capture relative importance of the one word with others - a concept called **`Attention`** used in Transformers

Another big change in Transformer architecture is ability to process the inputs in parallel - `Multi-Head` attention
<br>

Before we jump in to Transformer and it's architecture, we will cover some basics

# Basics

## 1. Average & Wt Average

Average is used to calculate mean & Weighted Average is used to improve data accuracy like shown below. <br>

<img width="396" alt="image" src="https://github.com/rvbug/NLP/assets/10928536/0edb77b7-7fbc-47e7-9ad3-a8916fde1711">
<br>

Context Vector is nothing but `Weighted sum of input vectors`

## 2. Context Vector / Thought Vector

Context Vector is a compact representation (in a fixed format) trying to capture the semantic meaning of the input sentences. 
Typically, all information (hidden states) from the **Encoder layer** creates these context vectors sending it to **Decoder layer** as inputs and generate output sequences. 

One of the major disadvtange is that the performance of these architecture drops drastically if the input sequences are very long

<img width="343" alt="image" src="https://github.com/rvbug/NLP/assets/10928536/b846c8eb-6b51-4cc8-8067-916921d2ca74"> 


## Attention 

Instead of encoding the input sequence into a single fixed context vector, is it possible to build a context vector for each output time step? 
This is basis of **"Attention"**

<img width="166" alt="image" src="https://github.com/rvbug/NLP/assets/10928536/0b4e6bfb-e9da-4722-a8a9-047aceae2e09">














<br><br><br><br><br>
If you look at the [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf) paper, the `Dot Product` attention uses the formula as below.  
Softmax function helps to attain non linearity and helps scaling weights between 0 & 1.  

<img width="194" alt="image" src="https://github.com/rvbug/NLP/assets/10928536/864cadcc-bdb4-4aef-a9d7-13e52489acf0">
<img width="404" alt="image" src="https://github.com/rvbug/NLP/assets/10928536/c0fe6f5b-e6e4-482f-ad28-66d307b46cd1">
<br>
`Note`: For efficency in calculation Stack all all Queries together & Keys together (np.vstack?)  
You can make the query, key and value vectors smaller using projection vectors via linear transformations.  
These Projections are learnable parameters ($W_q, W_k, W_v$)






## Self Attention
The idea is to add a contextual information to the words in a sentence. i.e. how important is that word to the others

## Masking

(The process of selective masking is called `Attention` but to create these masks are not a straight-forward)

Masking - Means To hide something. Keeping most of the values as  zero and multiply (element wise) it with the features will ensure we can attend to the most important word as the the rest of the values will be zero. This will help in masking the unwated feature and next word predictions will become very strong. 
This is called `Selective Masking`

## Transition Matrix  
Is a way to represent sequences of words i.e which words are before or after the current one. One of them is known as Markov chain. **They are the Keys (K)**.
The Markov chain is probabilities for the next word depending on the recent word(s)
  - If next word depend only on single most recent word - then it is `first order Markov Model`
  - If next word depend on two most recent words - then it is `second order Markov Model`



```python
import numpy as np

# q is the query i.e. the feature we are interested in
q = np.array([0,1, 0])
# k is the collection of mask also known as key in row format

k = np.array([[0,0,0], [0,1,1], [0,0,0]])

# dot product 
q @ k

# Going to query for 2nd row in the key matrix

#### output
# array([0, 1, 1])
```

So, the Mask Lookup is going to be used using  - $\(Q * K^T)$ . `T` is transpose since it will make this in Column format like so:

<img width="638" alt="image" src="https://github.com/rvbug/NLP/assets/10928536/ba3a9cc0-46b8-44ae-94c3-c3c3ad2e6f2a">



# Embeddings

Embeddings are nothing but projecting matrix from one space to another using matrix multiplication
Embedding helps representing words in any number of dimensions and is usually dense (vs sparse if you use one-hot encoding).

There are two types of embeddings `Word` (dealing with words) and `Sentence Embeddings` which is for an entire sentence

The Transformer models will be very large with over billion parameters if we do not use embeddings. Embeddings helps reduces the parameters.
The lower the parameters, the semantic meaning might not be captured with greater accuracy. Higher the parameters, the computation costs will be high.
This is the trade off.


# Positional Encoding

The input is tokenized and fed as a whole sentence so we need to keep the position of the token in the sequence hat is why we use Positional Encoding. 
This encoding is added with each token embedding.

<img width="322" alt="image" src="https://github.com/rvbug/NLP/assets/10928536/4ad3af82-0a0e-4a44-b805-cb46335b4ae7">








### FCNN
For building features, one can use the FCNN (*F*ully *C*onnected *N*eural *N*etwork)

`Note: In the Paper The encoder network has 2 section. First, is the Attention and Second is FCNN`

<img width="700" alt="image" src="https://github.com/rvbug/NLP/assets/10928536/decf4f74-c119-43d7-b619-59a5caf425b8">




<br>
<br>


## Multi-Head Attention

N = Vocab size (all the words in your corpus)
n = Maximum sequence length in (# of words with largest sentence) *[2048 in GPT-3]*
dim = Embedding Dimensions *(usually 512)*



## Context Vector

Content Vector **(\$C_i$)** for output **(\$Y_i$)** is generated using the weight sum of hidden states **(\$h_i$)** using the formula


**\$C_i = \sum{_i^n} \alpha{_i}{_j} * h_i $**

where \$n$ is number of words and \$\alpha{_i}{_j}$ is calculated using Softmax function

## Query & Keys
keys are the inputs 

## Values

## Encoder


## Decoder


<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
# References

[Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)  
[Jay Alammar's Blog](http://jalammar.github.io/illustrated-transformer/)    
[Positional Encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)     
[Deep Learning - Transformers](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html)  
