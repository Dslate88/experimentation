import math

import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer  # torchtext

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head  # number of heads
        self.d_model = d_model  # dimension of embedding model
        self.d_k = d_model // n_head # dimension of head

        self.W_Q = nn.Linear(d_model, d_model) # linear transformation for query
        self.W_K = nn.Linear(d_model, d_model) # linear transformation for key
        self.W_V = nn.Linear(d_model, d_model) # linear transformation for value


    def scaled_dot_product_attention(self, query, key, value):
        # calculate attention score
        score = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.d_k)

        # calculate weights
        attn_weights = torch.softmax(score, dim=-1)
        output = torch.matmul(attn_weights, value)
        return output, attn_weights


## test outputs for learning...

# test vector
text_vector = ["I am learning about multi-head attention", "What is the weather today?", "what is stability.ai up to?"]

# tokenize vector
tokenized_vector = get_tokenizer("basic_english")(text_vector[0])
print(tokenized_vector)
