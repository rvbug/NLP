# Basics

Masking - Means To hide something. Keeping most of the values as  zero and multiply (element wise) it with the features will ensure we can attend to the most important word as the the rest of the values will be zero. This will help in masking the unwated feature and next word predictions will become very strong. 
This is called `Selective Masking`

`Transition Matrix` - Is a way to represent sequences of words. One of them is known as Markov chain. They are the Keys (K).
The Markov chain is probabilities for the next word depending on the recent word(s)
  - If next word depend only on single most recent word - then it is `first order Markov Model`
  - If next word depend on two most recent words - then it is `second order Markov Model`
 


## Attention 

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



## Self Attention






# References
[Attention is all you need](https://)
