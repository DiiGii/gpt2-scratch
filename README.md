# gpt2-scratch

## Overview
Implementing GPT2 from scratch! Built with PyTorch, pandas, HuggingFace transformers library, and more!
- Access each Colab notebook to see code implementations of core functionalities 
- Read topic summaries for each Colab notebook below 👇

---


## Attention transformer (gpt2_attention_transformer.ipynb)

In the following notebook, we will be implementing an attention transformer in PyTorch. We'll accomplish this by implementing each of the following:

1. LayerNorm

  Layer Normalization (**LayerNorm**), used in GPT-2, normalizes activations within each training example across all its features, unlike Batch Normalization which normalizes across a batch. Applied after attention and feed-forward layers, LayerNorm stabilizes training by reducing internal covariate shift, leading to faster training and better performance.

2. Attention mechanism
  
  The **attention mechanism** is the heart of GPT2, because it solves the problem of long-range dependencies -- when words that are far apart are related. In particular, the attention mechanism lets the model directly consider every word when processing each one, like a spotlight highlighting relevant words regardless of their position.

3. Multi-head attention

  **Multi-head attention** enhances the basic attention mechanism by running multiple attention calculations in parallel. Each "head" learns different aspects of the input, like in a sentence, one head might focus on the action while another focuses on the subject's description. This parallel processing allows the model to capture more diverse information and semantic relationships, especially useful in longer sequences for improved accuracy and efficiency.

4. Bigram Attention Model

  The **BigramAttentionModel** uses token embeddings (representing word meaning) and positional embeddings (representing word order) combined as input to transformer blocks with multi-head attention. A final linear layer maps these to the vocabulary for text generation, with LayerNorm for regularization. Training uses cross-entropy loss, and generation is done by iteratively predicting characters. In this step, we put together all of our work in the previous steps and test our model. 

## Stochastic Decoding (gpt2_stochastic_decoding.ipynb)

In GPT2, **stochastic decoding** is a way of introducing randomness into text generation. 

Why would we need randomness? Well, when the model always picks the single likeliest word to come next (**greedy decoding**), the quality of the text often suffers. Greedy decoding often leads to repetitive and predictable text, which is why to introduce some creativity and get higher quality responses, we use stochastic decoding. 

In the code below, we will explore 3 stochastic decoding methods:
- **Temperature scaling**: This involves adjusting the probability distribution of the next word by a "temperature" parameter. A higher temperature makes the distribution flatter, giving more weight to less likely words, while a lower temperature makes the distribution sharper, favoring more likely words.
- **Top-k sampling**: This involves selecting the k most likely next words and then sampling from them according to their probabilities. This limits the model's choices to the most promising candidates while still introducing some randomness. 
- **Top-p (nucleus) sampling**: This is similar to top-k sampling, but instead of selecting a fixed number of words, it selects the smallest set of words whose cumulative probability exceeds a threshold p. This dynamically adjusts the number of candidates based on the probability distribution. 
