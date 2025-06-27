#Importing necessary Libraries

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy, pandas
import matplotlib.pyplot as plt
import math


#OneHot Embedding
class OneHotEmbed(nn.Module):

  def __init__(self, vocab_size: int):                           #vocab_size is S here (if RPE is done, if not S+T to encode positions)
    super().__init__()
    self.vocab_size=vocab_size

  def forward(self, x):
    return F.one_hot(x, num_classes= self.vocab_size).float()    #COnvert to float to support gradient backpropagation

#Relative Positional Embedding
class RPE(nn.Module):

  def __init__(self, max_position:int ):                                                      #Here Max position is M
    super().__init__()
    self.max_position = max_position
    self.relative_embedding = nn.Embedding(max_position, 1)                                   #Look up table for embeddings in each position

  def forward(self, seq: int):
    dist_matrix=torch.zeros((seq,seq))
    for i in range(seq):
      for j in range(seq):
        dist_matrix[i][j]= j-i

    embedding_indices = (-dist_matrix.clamp(min=-self.max_position, max=-1)).long() - 1      # This maps -1=> 0, -2=> 1 and so on to navigate the look up table

    
    embedding_indices=embedding_indices.to(device='cuda')
    #Zero out future positions and beyond max_position (inf before softmax)
    mask=(dist_matrix.abs()>=self.max_position) | (dist_matrix>=0)
    RP_embed=self.relative_embedding(embedding_indices)                                      #navigates to the look up table and replaces it with embedding learnt
    RP_embed[mask]=-1e9

    return RP_embed


#Layernorm
class Layernorm(nn.Module):

  def __init__(self, features: int, eps: float = 1e-6):
    super().__init__()
    self.alpha = nn.Parameter(torch.ones(features))
    self.beta= nn.Parameter(torch.zeros(features))
    self.eps = eps

  def forward(self, x):
    #x.shape = (batch, seq, hidden_size)
    mean = x.mean(-1, keepdim=True) #(batch, seq, 1)
    std = x.std(-1, keepdim=True)   #(batch, seq, 1)

    x = self.alpha*(x-mean)/ (self.eps+ std) + self.beta
    return x

#Disentangled Multi-head Self Attention
class Disentangled_MHSA(nn.Module):

  def __init__(self, d_model:int, num_heads:int, max_position: int, dropout:float):       #d_model = S
    super().__init__()
    self.d_model=d_model                                                                  #Embedding vector size - subject to change in each layer
    self.num_heads=num_heads                                                              #number of heads - subject to change in each layer
    self.RPE = RPE(max_position)

    self.w_q = nn.Linear(d_model, num_heads*d_model, bias=False)
    self.w_v= nn.Linear(d_model, num_heads*d_model, bias= False)
    self.w_k = nn.Linear(d_model, num_heads* d_model, bias=False)

    #No output projection matrix here becasue we are concatentating the heads

    #self.dropout = nn.Dropout(dropout) - Check if we need dropout here ?


  def attention(self, query, key, value, dropout:nn.Dropout):                             # Mask : Causal attention
    d_k = query.shape[-1]
    seq = query.shape[-2]
    attention_scores = (query @ key.transpose(-2,-1))/math.sqrt(d_k)                      # dimension = (batch_size, num_heads, T , T)

    #applying causal mask
    mask=torch.zeros((seq,seq)).to(device='cuda')
    for i in range(seq):
      for j in range(seq):
        if (j<=i):
          mask[i][j]= 1
    attention_scores = attention_scores.masked_fill_(mask==0, -1e9)


    #Adding RPE here
    rpe=self.RPE(seq)
    #print(attention_scores.shape)
    rpe=rpe.to(attention_scores.device)
    rpe=rpe.squeeze(-1)
    rpe=rpe.unsqueeze(0).unsqueeze(0)
    #print(rpe.shape)
    attention_scores_rpe=attention_scores+rpe

    #softmax
    attention_scores_softmax = attention_scores_rpe.softmax(dim=-1)

    #Should we drop out attention score ?
    return (attention_scores_softmax@value), attention_scores_rpe                                    #(batch, num_heads, T, S)


  def forward(self, q,k,v):
    query = self.w_q(q)                                                                   #q = (batch, T, S) --> query = (batch, T, num_heads*S)
    key = self.w_k(k)
    value = self.w_v(v)

    query =query.view(query.shape[0], query.shape[1], self.num_heads,self.d_model).transpose(1,2)       #(batch, T, num_heads*S) -->  (batch, T, num_heads, S) --> (batch, num_heads, T,  S)
    key = key.view(key.shape[0], key.shape[1], self.num_heads,self.d_model).transpose(1,2)
    value = value.view(value.shape[0], value.shape[1], self.num_heads,self.d_model).transpose(1,2)
    output, self.attention_scores = self.attention(query, key, value, dropout =0.0)        #x.shape = (batch, num_heads, T, S)

    #concatenating to return
    output = output.permute(0, 2, 1, 3)                                                                 #x.shape = (batch, T, num_heads, S)
    output = output.reshape(output.shape[0], output.shape[1], self.num_heads*self.d_model)              #x.shape = (batch, T, num_heads*S)
    #print('attention output', output.shape)
    return output

#Residual Connection
class ResConnect(nn.Module):

  def __init__(self, features: int, dropout: float):
    super().__init__()
    self.dropout=nn.Dropout(dropout)
    self.layernorm=Layernorm(features)                #Callin pre-defined class

  def forward(self, x, sublayer):
    output= self.dropout(sublayer(self.layernorm(x)))    # Note: Pre-norm is applied before passing it through the layer for training stability
    #print(output.shape)
    #print(x.shape)
    return torch.cat([x, output], dim=-1)                #Concatenated output

#Induction Transformer
class InductionHead(nn.Module):

  def __init__(self, d_model, num_heads1, num_heads2, max_position, vocab_size):           #d_model = S(one hot), vocab_size=S
    super().__init__()
    self.attn1 = Disentangled_MHSA(d_model, num_heads1, max_position, dropout=0.0)
    self.attn2 = Disentangled_MHSA(d_model*(1+num_heads1), num_heads2, max_position,dropout=0.0)
    self.res1 = ResConnect(d_model, dropout=0.0)
    self.res2 = ResConnect(d_model*(1+num_heads1), dropout=0.0)
    self.w_o = nn.Linear(d_model*(1+num_heads1)*(1+num_heads2), vocab_size)

  def forward(self, x):
    #print(x.shape)
    x = self.res1(x, lambda x_: self.attn1(x, x, x))                               #(batch, T , S*(1+num_heads1))
    #print(x.shape)
    x = self.res2(x, lambda x_: self.attn2(x, x, x))                               #(batch, T , S*(1+num_heads1)*(1+num_heads2))
    #print(x.shape)
    logits = self.w_o(x)                                                                 #(batch, T, S)
    #probabs = F.softmax(logits, dim=-1)
   # prediction = probabs.agrmax(dim=-1)                                                  #(batch, T)
    return logits

