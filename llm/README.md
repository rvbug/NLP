# Introduction

What are *L*arge *L*anguage *M*odels - LLMs are used to process large amount of text. It is 'Large' because it is trainined on huge datasets using Deep Neural Networks.


# Sentence Embeddings
While word embeddings are useful but they do not take into consideration the position or order of the words. This is the concept used in Sentence Embedding.
So instead of transforming words to number, the entire sentence is converted to numbers. Sentence embeddings are very powerful in the sense that vectors are assigned to each sentences in such a way that every word and it's positions carry importance.

```python
pip install -U sentence-transformers
```

In the below example you will see the sentence embedding in action. The output of this model is (3, 384) where 3 is the number of sentences and 384 is representation of each sentences. You can call it 384 features.


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

Once the embedding vectors are generated, you can find the similarity using dot product. If the dot product between 2 vector are greater, the sentences will be similar.








<br>
<br>
# References  

[SBERT - Sentence Transformers](https://www.sbert.net/)
