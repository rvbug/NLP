# Transformers
RNNs had problems dealing with large sentences and LSTMs were slow and sequential. This was solved by using Transformers which works on the whole sentences and some parts of the arichitecture can run in parallel. 




# Basics

Masking - Means To hide something. Keeping most of the values as  zero and multiply (element wise) it with the features will ensure we can attend to the most important word as the the rest of the values will be zero. This will help in masking the unwated feature and next word predictions will become very strong. 
This is called `Selective Masking`

`Transition Matrix` - Is a way to represent sequences of words. One of them is known as Markov chain. They are the Keys (K).
The Markov chain is probabilities for the next word depending on the recent word(s)
  - If next word depend only on single most recent word - then it is `first order Markov Model`
  - If next word depend on two most recent words - then it is `second order Markov Model`
 


## Attention 

## Embeddings
The Transformer models will be very large with over billion parameters if we do not use embeddings.  Embeddings helps reduces the parameters.
The lower the parameters, the semantic meaning might not be captured with greater accuracy. Higer the parameters, the computation costs will be high.
This is the trade off.

Embeddings are nothing but projecting matrix from one space to another using matrix multiplication
Embedding helps representing words in any number of dimensions and is usually dense (vs sparse if you use one-hot encoding).

There are two types of embeddings Word (1-Hot, W2V etc) - which deals with words and Sentence Embeddings which is for an entire sentence




### Masking 

The process of selective masking is called `Attention` but to create these masks are not a straight-forward. 
In Transformers, the mask is generated via techniques which will be discussed shortly.


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

So, the Mask Lookup is going to be used using  - $\(Q * K^T)$  
`T` is transpose since it will make this in Column format as shown below

<img width="638" alt="image" src="https://github.com/rvbug/NLP/assets/10928536/ba3a9cc0-46b8-44ae-94c3-c3c3ad2e6f2a">

### Building Features
For building features, one can use the FCNN (*F*ully *C*onnected *N*eural *N*etwork)

`Note: In the Paper The encoder network has 2 section. First, is the Attention and Second is FCNN`

<img width="700" alt="image" src="https://github.com/rvbug/NLP/assets/10928536/decf4f74-c119-43d7-b619-59a5caf425b8">





## Positional Encoding
<br>
<br>
<br>
<br>
<br>
<br>
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
