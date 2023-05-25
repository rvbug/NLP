# Introduction

RNNs had problems dealing with large sentences and LSTMs were slow and sequential. This was solved by using Transformers. It can work on the entire sentences (instead of words) and some parts of the arichitecture even can run in parallel. 

Transformers uses encoder-decoder architecture :-

<img width="317" alt="image" src="https://github.com/rvbug/NLP/assets/10928536/d72433cb-4ef7-48e2-9aca-9f4923a7b6ad">

# Basics

## 1. Average & Wt Average
Average - To calculate mean & Weighted Average is used to improve data accuracy   

<img width="396" alt="image" src="https://github.com/rvbug/NLP/assets/10928536/0edb77b7-7fbc-47e7-9ad3-a8916fde1711">

## 2. Context Vector / Thought Vector
Was first used in `seq-to-seq` models where all the inputs are generated in the **Encoder layer** and is represented in a fixed format known as Context Vector.
It contains all the information from the hidden state of Encoder Layer.

This vector is then passed to the **Decoder layer** as inputs to generate output sequence. You can think of Context Vector as compact representation of the inputs capturing the  semantic meaning. Disadvtange - If the sequences are very long then performance drops drastically as the input information can be lost.

Instead of encoding input to fixed sized context vector, what can be done? 

So Context Vector is noting but `Weighted sum of input vectors`


## Query & Keys


## Values



## Attention 



Instead of encoding the input sequence into a single fixed context vector, can we build a context vector for each output time step?    
Yes, this is known as Attention (A retrival process which uses weighted average of values).   

Attention can be calculated by performing the dot-product (MatMul) of the dynamically attended weights with the input sequence (V).

Where attended Weights = `np.dot(Q, K)`

<img width="194" alt="image" src="https://github.com/rvbug/NLP/assets/10928536/864cadcc-bdb4-4aef-a9d7-13e52489acf0">
<img width="404" alt="image" src="https://github.com/rvbug/NLP/assets/10928536/c0fe6f5b-e6e4-482f-ad28-66d307b46cd1">
<br>

`Softmax` helps in non linearity and also to scale weights between 0 & 1.

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





<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
# References

[Attention is all you need](https://)  
[Jay Alammar's Blog](http://jalammar.github.io/illustrated-transformer/)   
[Positional Encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)   
