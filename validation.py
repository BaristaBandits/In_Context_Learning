
import torch
#importing the Induction transformer
from model import OneHotEmbed, InductionHead
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy, pandas
import matplotlib.pyplot as plt
import math

class Bigram:
  def __init__(self, vocab_size, T):
    self.vocab_size=vocab_size
    self.T=T

  def __call__(self, x):
    batch_size, T= x.shape
    bigram_counts=torch.zeros((batch_size, T, self.vocab_size), device='cuda')
    for t in range(0, T):
      prefix = x[:, :t+1]
      for b in range(batch_size):
        current_state=prefix[b, -1]
        following_tokens=[]
        for i in range(t):
          if (prefix[b, i]==current_state):
            following_tokens.append(prefix[b, i+1])
        counts=torch.bincount(torch.tensor(following_tokens, dtype=torch.long), minlength=self.vocab_size).float()
        total=counts.sum()
        bigram_counts[b, t]= (counts+1)/(total+ 5)  if total > 0 else counts
    return bigram_counts


class MarkovChain:
  
    def __init__(self, order, vocab_size, device, transition_matrix=None, start = None, t_matrix_in_context = False,  dirichlet_alpha=None):
        self.order = order
        self.vocab_size = vocab_size
        self.device = device
        self.start = start if start is not None else order                #How many steps to initialize before starting transitions?
        self.t_matrix_in_context = t_matrix_in_context
        self.dirichlet_alpha = dirichlet_alpha if dirichlet_alpha is not None else torch.ones(vocab_size, device=device)
        if not t_matrix_in_context:
            if transition_matrix is None:
                # Generate a random stochastic matrix
                raw_matrix = torch.rand(vocab_size, vocab_size, device=device)
                self.transition_matrix = raw_matrix / raw_matrix.sum(dim=1, keepdim=True)
            else:
                self.transition_matrix = transition_matrix.to(device)


        # Compute the stationary distribution using eigenvalues and eigenvectors
        if transition_matrix is not None:
            eigvals, eigvecs = torch.linalg.eig(self.transition_matrix.t())
            eigvals = eigvals.real  # Considering only the real parts
            stationary_index = torch.argmin(torch.abs(eigvals - 1))
            self.stationary = eigvecs[:, stationary_index].real
            self.stationary /= self.stationary.sum()  # Normalize
            self.stationary = self.stationary.to(device)
            self.entropy_stationary = -torch.sum(self.stationary * torch.log(self.stationary))
            self.entropy_rate = self.calculate_entropy_rate()
        else: # No transition matrix provided
            self.stationary = None
            self.entropy_stationary = None
            self.entropy_rate = None
            self.transition_matrix = None

    def calculate_entropy_rate(self):
        log_p = torch.log(self.transition_matrix)
        mu_p_log_p = self.stationary.unsqueeze(0).T * self.transition_matrix * log_p
        entropy_rate = -torch.sum(mu_p_log_p)
        return entropy_rate.item()

    def get_batch(self, seq_length, batch_size, initial='steady', return_ttensor=False):
        data = torch.zeros(batch_size, seq_length + 1, device=self.device, dtype=torch.int64)
        if self.t_matrix_in_context:
            #sample on transition matrix per batch poin such that each chain has its own transition matrix, sample each row of the transition matrix according to the dirichlet distribution
            transition_tensors = torch.distributions.dirichlet.Dirichlet(self.dirichlet_alpha).sample((batch_size,self.vocab_size))
            # if initial was steady print a waring that for the in-context task it is not implemented so it will be set to uniform
            #if initial == 'steady':
                #print('Warning: In-context transition matrix is not implemented for steady initial distribution. Setting initial to uniform.')
            initial_distribution = torch.full((self.vocab_size,), 1.0 / self.vocab_size, device=self.device)
        else:
            transition_tensors = None
            if initial == 'steady':
                initial_distribution = self.stationary
            elif initial == 'uniform':
                initial_distribution = torch.full((self.vocab_size,), 1.0 / self.vocab_size, device=self.device)

        data[:, :self.start] = torch.multinomial(initial_distribution, batch_size * self.start, replacement=True).view(batch_size, self.start)

        for i in range(self.start, seq_length + 1):
            data[:, i] = self.get_next_symbols(data[:, i-self.order], transition_tensors)

        if return_ttensor:
            return data[:, :seq_length], transition_tensors
        else:
            return data[:, :seq_length]

    def get_next_symbols(self, prev_state, transition_tensors=None):
        if self.t_matrix_in_context:
            probs = transition_tensors[torch.arange(prev_state.size(0)), prev_state, :]
        else:
            probs = self.transition_matrix[prev_state]
        next_state = torch.multinomial(probs, num_samples=1).squeeze(1)
        return next_state


order = 1
S = 5           #Number of states in the markov chain
vocab_size = S
device = 'cuda'
batch_size=100
T = 128          #Length of Markov Chain
M = 128

mc = MarkovChain(order, vocab_size, device, t_matrix_in_context=True)
data = mc.get_batch(seq_length=T+1, batch_size=batch_size, initial='steady')
src_data = data[:, :-1]  # (batch, T) - we are resitrciting it till T-1 steps
tgt_data = data[:, 1:]   # (batch, T) — next tokens from 1 to T (left shifted)



from model import InductionHead                                  #pre-defined model

induction_transformer=InductionHead(S,2,2,M,S)                   #Initialization
criterion=nn.CrossEntropyLoss()                                  #Loss Function
#optimizer=optim.Adam(induction_transformer.parameters(), lr=lr)    #Optimizer
checkpoint = torch.load("checkpoint.pt", map_location="cuda" if torch.cuda.is_available() else "cpu")
print(checkpoint.keys())

induction_transformer.load_state_dict(checkpoint["model_state_dict"])
induction_transformer.to(device)  # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
induction_transformer.eval()
train_losses = []

bigram=Bigram(S,T)(src_data)
print(bigram.shape)
print(tgt_data.shape)
src_data = OneHotEmbed(S)(src_data)
output=induction_transformer(src_data)
loss_transformer= criterion(
        output.contiguous().view(-1, S),      # logits
        tgt_data.contiguous().view(-1)        # true next token
)
loss_bigram= criterion(
        bigram.contiguous().view(-1, S),      # logits
        tgt_data.contiguous().view(-1)        # true next token
)
print(f"Transformer Loss:{loss_transformer.item()}")
print(f"Bigram Loss:{loss_bigram.item()}")
