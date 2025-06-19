#importing the Induction transformer
from model import OneHotEmbed, InductionHead
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy, pandas
import matplotlib.pyplot as plt
import math

#Generating the data
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


#Hyperparameters
T = 128          #Length of Markov Chain
S = 5           #Number of states in the markov chain
M = 128          #Max position for RPE

num_epochs = 10000
lr=0.001
batch_size = 64
order = 1
vocab_size = S
device = 'cpu'


induction_transformer=InductionHead(S,2,2,M,S)                   #Initialization
criterion=nn.CrossEntropyLoss()                                  #Loss Function
optimizer=optim.Adam(induction_transformer.parameters(), lr=lr)    #Optimizer

induction_transformer.train()
train_losses = []

plt.ion()  # Enable interactive mode
fig, ax = plt.subplots()
line, = ax.plot(train_losses)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')




mc = MarkovChain(order, vocab_size, device, t_matrix_in_context=True)

for epoch in range(num_epochs):
  data = mc.get_batch(seq_length=T+1, batch_size=batch_size, initial='steady')
  src_data = data[:, :-1]  # (batch, T) - we are resitrciting it till T-1 steps
  tgt_data = data[:, 1:]   # (batch, T) — next tokens from 1 to T (left shifted)
  src_data = OneHotEmbed(S)(src_data)
  optimizer.zero_grad()
  output=induction_transformer(src_data)
  if epoch==0:
    print(output.shape, tgt_data.shape)
    ax.set_xlim(0, 100)
  loss = criterion(
        output.contiguous().view(-1, S),      # logits
        tgt_data.contiguous().view(-1)        # true next token
    )
  loss.backward()
  optimizer.step()
  if epoch%100==0:
    print(f"Iteration:{epoch}. Loss:{loss.item()}")
  train_losses.append(loss.item())
  line.set_ydata(train_losses)
  line.set_xdata(range(len(train_losses)))
  ax.relim()
  ax.autoscale_view()
  fig.canvas.draw_idle()
  plt.pause(0.1)  # Pause to update plot

  #print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

plt.ioff()
plt.show()
