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

## 3. Embeddings

Embeddings are nothing but projecting matrix from one space to another using matrix multiplication
Embedding helps representing words in any number of dimensions and is usually dense (vs sparse if you use one-hot encoding).

There are two types of embeddings `Word` (dealing with words) and `Sentence Embeddings` which is for an entire sentence

The Transformer models will be very large with over billion parameters if we do not use embeddings. Embeddings helps reduces the parameters.
The lower the parameters, the semantic meaning might not be captured with greater accuracy. Higher the parameters, the computation costs will be high.
This is the trade off.


# Attention 

Instead of encoding the input sequence into a single fixed context vector, is it possible to build a context vector for each output time step? 
This is basis of **"Attention"**

Most simplified version of the attention is as shown below.
Typically, the input will be tokenized and run through embedding layer helping to understand context of words.
The output of the emebedding is now multiplied with a some weighing factor to generate output which has lot more context. 
<br>
<img width="1047" alt="image" src="https://github.com/rvbug/NLP/assets/10928536/31c649a5-8d70-4a2f-a101-4b979d499468">
<br>

If we now have to have a contextualised representation of 3rd vector then : <br>
<img width="182" alt="image" src="https://github.com/rvbug/NLP/assets/10928536/e17f33a6-9b0f-4a96-b346-ca3995883e9d">
<br>
The blue dotted lines - that is `Attention` for you. You can think of it as below <br>
<img width="126" alt="image" src="https://github.com/rvbug/NLP/assets/10928536/ccd03cf7-e0d2-42bf-979b-61a8f4c008ae">
<br>


# Self Attention
The idea is to add a contextual information to the words in a sentence. i.e. how important is that word to the others.  
Input Embedding vectors \$v = [v_0, v_1...v_n]$ is passed through \$W_Q, W_K, W_V$ which are Matrix for Query, Key and Value respectively.

These matrix can be learnt through back propogation (follow the while lines). The entire block is called self attention 
<br>
<img width="414" alt="image" src="https://github.com/rvbug/NLP/assets/10928536/0db0a9ec-739a-4195-bc1f-251c281d3e06">
<br>
You can stack any number of **Self Attention** block like so <br>
<img width="185" alt="image" src="https://github.com/rvbug/NLP/assets/10928536/07e64c70-c71b-4343-aa23-f79fec596bf7">

## Multi-Head Attention

N = Vocab size (all the words in your corpus)
n = Maximum sequence length in (# of words with largest sentence) *[2048 in GPT-3]*
dim = Embedding Dimensions *(usually 512)*



<br>
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
<br>


# References
[Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)  
[Jay Alammar's Blog](http://jalammar.github.io/illustrated-transformer/)    
[Positional Encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)     
[Deep Learning - Transformers](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html)  
