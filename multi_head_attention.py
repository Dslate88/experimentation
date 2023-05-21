import torch.nn as nn
from torchtext.data.utils import get_tokenizer  # torchtext

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head  # number of heads
        self.d_model = d_model  # dimension of model
        self.d_k = d_model // n_head # dimension of head

        self.W_Q = nn.Linear(d_model, d_model) # linear transformation for query
        self.W_K = nn.Linear(d_model, d_model) # linear transformation for key
        self.W_V = nn.Linear(d_model, d_model) # linear transformation for value

# test vector
text_vector = "I am learning about multi-head attention. "

# tokenize vector
tokenized_vector = get_tokenizer("basic_english")(text_vector)
print(tokenized_vector)


