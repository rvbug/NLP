# Introduction

Before understanding what Transformers are, it is important to understand what `Attention` & `Self Attention` mechanisms.  
This concept made the Transformers one of the biggest breakthroughs in Deep Learning and rise of lot of AI tools.

As humans, we tend to concentrate on the things that matters the most i.e. we pay more attention to the things which are interesting. 
The same concept can be applied to machines.

Attention was first used  `Encoder-Decoder` Architecture where the final output of RNN/LSTM creates a **Context Vector**. 
If sentences were longer, these vectors could not capture semantic meaning. 
RNNs also had it's share of problems dealing with large sentences while LSTMs were slow inputs are processed sequentially. <br>

So, capture relative importance of the one word with others in a context vectors  -  known as **`Attention`**   
Another big change in Transformer architecture is ability to process the inputs in parallel using `Multi-Head` attention
<br>

Before we jump in to Transformer and it's architecture, we will cover some basics

# 1. Basics

## 1.1 Average & Wt Average

Average is used to calculate mean & Weighted Average is used to improve data accuracy. <br>

<img width="400" alt="image" src="https://github.com/rvbug/NLP/assets/10928536/0fda4c8b-9221-4fbd-bb91-a5b32fd4ba64">

<br>

[Check out the code here](https://github.com/rvbug/NLP/blob/main/transformers/weighted_avg.py)

<br>
Context Vector is nothing but `Weighted sum of input vectors`

## 1.2 Context Vector / Thought Vector

Context Vector is a compact representation (in a fixed format) trying to capture the semantic meaning of the input sentences. 
Typically, all information (hidden states) from the **Encoder layer** creates these context vectors sending it to **Decoder layer** as inputs and generate output sequences. 

One of the major disadvtange is that the performance of these architecture drops drastically if the input sequences are very long

<img width="383" alt="image" src="https://github.com/rvbug/NLP/assets/10928536/6e044aaf-1921-43b5-8559-fb78e847986c">

Instead of encoding the input sequence into a single fixed context vector, is it possible to build a context vector for each output time step?


## 1.3 Linear Layer

If output of NN is a linear function of the inputs then this layer is called a `Linear Layer`. In other words it can perform linearly transformations on the input data.   
\$o/p = f(input)$

In the attention mechanism, the Linear layer does not have a bias term but only weights.   
Note: if we design a good NN, it can easily understand more complex non-linear functions.   


## 1.4 Tokenize & Embeddings

Embeddings are nothing but projecting matrix from one space to another using matrix multiplication
Embedding helps representing words in any number of dimensions and is usually dense (vs sparse if you use one-hot encoding).

There are two types of embeddings `Word` (dealing with words) and `Sentence Embeddings` (for entire sentence)

The Transformer models will be very large with over billion parameters if we do not use embeddings. Embeddings helps reduces the parameters.
The lower the parameters, the semantic meaning might not be captured with greater accuracy. Higher the parameters, the computation costs will be high.
This is the trade off.

```python
embedding_layer = tf.keras.layers.Embedding(3, 5) # (vocab_size+1, feature)
result = embedding_layer(tf.constant([1, 2])) # (2,)
result.shape # TensorShape([2, 5]))
```

[See the sample code here](https://github.com/rvbug/NLP/blob/main/transformers/token_embeddings.py)

## 1.5 Sentence Embeddings

While word embeddings are useful but they do not take into consideration the position or order of the words, a concept used in Sentence Embedding.
So instead of transforming words to number, the entire sentence is converted to numbers. Sentence embeddings are very powerful in the sense that vectors are assigned to each sentences in such a way that every word and it's positions carry importance.

```python
pip install -U sentence-transformers
```

In the below example you will see the sentence embedding in action. The output of this model is (3, 384) where 3 is the number of sentences and 384 is representation of each sentences. Think 384 features.


```python
# import sentence-transformers & load a model
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

#sentences
sentences = ['this is for sentence transformes',
    'LLM is known as Lage Language Models',
    'Enjoy learning Deep Learning']

# encode the sentences
embeddings = model.encode(sentences)
# print shape of the emebeddings
print(embeddings.shape) # (3, 384)

```
Once the embedding vectors are generated, you can find the similarity using dot product. If the dot product between 2 vector are greater, those 2 sentences are similar.


# 2. Attention 

Look at 2 sentences below. What you will see is the use of bank in different context. As human, it is easy for us to determine the usage based on the neighbouring words. For machine, we need to do the same.

```python
s1 = "Bank of the river"
s2 = "money in the Bank"
```
Now calculate the similarity of each words with others and then if you are interested in "Bank", "money", "river", then see the weights created as below:

<img width="616" alt="image" src="https://github.com/rvbug/NLP/assets/10928536/efa45ef0-dff2-495b-9456-f941c5603ae1">


Another simplified version of the attention is  shown below.
Typically, the input will be tokenized and run through embedding layer helping to understand context of words.
The output of the emebedding is now multiplied with a some weighing factor to generate output which has lot more context. 
<br>
<img width="1044" alt="image" src="https://github.com/rvbug/NLP/assets/10928536/79e99cf0-e511-4063-b29c-9fd38113b761">

<br>

If we now have to have a contextualised representation of 3rd vector then : <br>
<img width="182" alt="image" src="https://github.com/rvbug/NLP/assets/10928536/e17f33a6-9b0f-4a96-b346-ca3995883e9d">
<br>
The blue dotted lines - that is `Attention` for you. You can think of stacking all the attention block <br>
<img width="126" alt="image" src="https://github.com/rvbug/NLP/assets/10928536/ccd03cf7-e0d2-42bf-979b-61a8f4c008ae">
<br>

# 3. Self Attention
The idea is to add a contextual information to the words in a sentence. i.e. how important is that word to the others.  
Input Embedding vectors \$v = [v_0, v_1...v_n]$ is passed through \$W_Q, W_K, W_V$ which are Matrix for Query, Key and Value respectively.

These matrix can be learnt through back propogation (follow the while lines). The entire block is called self attention 
<br>
<img width="300" alt="image" src="https://github.com/rvbug/NLP/assets/10928536/3ffbe186-2412-4b38-b620-b81beae59db0">

<br>
You can stack any number of **Self Attention** block like so <br>
<img width="200" alt="image" src="https://github.com/rvbug/NLP/assets/10928536/53315e26-5043-489f-bd8f-68975449a4a7">


# 4. Multi-Head Attention

Imagine a sentence having multiple attention, is there a way to parallelize it? We could use this idea to make the attention mechanism very efficient.<br>
<img width="339" alt="image" src="https://github.com/rvbug/NLP/assets/10928536/6fa756f5-ec9f-4647-b463-66a12012fdf6">
<br>

### Stacking Multi-headed attention models <br>
<img width="152" alt="image" src="https://github.com/rvbug/NLP/assets/10928536/3f46263c-9eb3-4462-95f4-7dd487540f90">
<br>

# 5. Encoder Block
In the paper, you would see a block for both Encoder and Decoder as it was specifically used for Language Translation task. But since we are focusing only on NLP skipping the Decoder will make sense. Here's how the Encoder looks like and how they can be stacked

<br>
<img width="466" alt="image" src="https://github.com/rvbug/NLP/assets/10928536/776c6b27-aa01-46d3-a64f-087beaa47e5d">
<br>

# 6. Positional Encoding

The input is tokenized and fed as a whole sentence so track the position of the token in the sequence, we use Positional Encoding. 
Encoding is added on top of the embedding.

Positional Vectors can be added to the Embedding vector like so.   
<br>
\$P_0 =$  \$f(i, n)$ <br>
<img width="350" alt="image" src="https://github.com/rvbug/NLP/assets/10928536/826aa09a-5a1b-402e-bce5-5f0e36310746">

# Next Step

Transformer Achitectures are the basic building block of LLMs (Large Language Models). You are now ready to learn "state-of-the-art" tools
Happy Learning!!


# References
[Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)  
[Jay Alammar's Blog](http://jalammar.github.io/illustrated-transformer/)    
[Positional Encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)     
[Deep Learning - Transformers](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html)  
[SBERT - Sentence Transformers](https://www.sbert.net/)
