
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
import wandb


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
        bigram_counts[b, t]= (counts+1)/(total+ 5)
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
batch_size=1000
T = 128          #Length of Markov Chain
M = 128

wandb.init(
    project="induction-transformer",  # change this to your project name
    name="bigram-vs-transformer-eval",  # optional: a specific run name
    config={
        "vocab_size": S,
        "order": order,
        "T": T,
        "batch_size": batch_size,
        "embedding_dim": M
    }
)


mc = MarkovChain(order, vocab_size, device, t_matrix_in_context=True)
data, transition_matrix = mc.get_batch(seq_length=T+1, batch_size=batch_size, initial='steady', return_ttensor=True)
src_data = data[:, :-1]  # (batch, T) - we are resitrciting it till T-1 steps
tgt_data = data[:, 1:]   # (batch, T) — next tokens from 1 to T (left shifted)
#print(transition_matrix.shape)


from model import InductionHead                                  #pre-defined model

induction_transformer=InductionHead(S,2,2,M,S)                   #Initialization
criterion=nn.CrossEntropyLoss()                                  #Loss Function
#optimizer=optim.Adam(induction_transformer.parameters(), lr=lr)    #Optimizer
checkpoint = torch.load("checkpoint.pt", map_location="cuda" if torch.cuda.is_available() else "cpu")
#print(checkpoint.keys())

induction_transformer.load_state_dict(checkpoint["model_state_dict"])
induction_transformer.to(device)  # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
induction_transformer.eval()
#train_losses = []

bigram=Bigram(S,T)(src_data)
#print(bigram.shape)
#print(tgt_data.shape)
src_data_onehot = OneHotEmbed(S)(src_data)
output=induction_transformer(src_data_onehot)
probs=F.softmax(output, dim=-1)

#cross entropy loss
loss_transformer= criterion(
        output.contiguous().view(-1, S),      # logits
        tgt_data.contiguous().view(-1)        # true next token
)
loss_bigram= criterion(
        bigram.log().contiguous().view(-1, S),      # logits
        tgt_data.contiguous().view(-1)        # true next token
)
print(f"Transformer Loss:{loss_transformer.item()}")
print(f"Bigram Loss:{loss_bigram.item()}")


#KL Divergence
def KL_Divergence(transition_matrix, src_data, transformer_probs, bigram_probs):

  batch_size, T, vocab_size = transformer_probs.shape
  gt_probs = torch.zeros_like(transformer_probs)

  for b in range(batch_size):
    for t in range(T):
      current_state=src_data[b,t].item()
      gt_probs [b, t]= transition_matrix[b, current_state]
  
  # To prevent division by zero
  epsilon = 1e-9
  P = gt_probs + epsilon
  Q = transformer_probs + epsilon
  R = bigram_probs + epsilon

  #KL(Q||P)
  kl_transformer= (P* (P.log () - Q.log())).sum(dim=-1)
  #KL(R ||P)
  kl_bigram= (P* (P.log () - R.log())).sum(dim=-1)
  #KL(Q||R)
  kl_transformer_bigram= (R* (R.log () - Q.log())).sum(dim=-1)
  
  plt.plot(range(T), kl_transformer.mean(dim=0).cpu().detach().numpy(), label='KL GT || Transformer')
  plt.plot(range(T), kl_bigram.mean(dim=0).cpu().detach().numpy(), label='KL GT || Bigram')
  plt.plot(range(T), kl_transformer_bigram.mean(dim=0).cpu().detach().numpy(), label='KL Bigram || Transformer')
  plt.xlabel('Time Step')
  plt.ylabel('KL Divergence')
  plt.legend()
  plt.grid(True)
  plt.savefig("KL_divergence_plot.png")

  # Log plot image
  wandb.log({"KL Divergence Plot": wandb.Image("KL_divergence_plot.png")})
  # Optional: Log each point as a line chart
  wandb.log({
        "KL over Time": wandb.plot.line_series(
            xs=list(range(T)),
            ys=[
                kl_transformer.mean(dim=0).cpu().detach().numpy(),
                kl_bigram.mean(dim=0).cpu().detach().numpy(),
                kl_transformer_bigram.mean(dim=0).cpu().detach().numpy()
            ],
            keys=["GT||Transformer", "GT||Bigram", "Bigram||Transformer"],
            title="KL Divergence vs Time",
            xname="Time Step t"
        )
    })
  
  return kl_transformer.mean(), kl_bigram.mean(), kl_transformer_bigram.mean()

print('\nKL Divergence Results \n')
#print("src_data", src_data.shape)
kl_transformer, kl_bigram, kl_transformer_bigram = KL_Divergence(transition_matrix, src_data, probs, bigram)
print(f"KL Divergence Transformer:{kl_transformer.item()}")
print(f"KL Divergence Bigram:{kl_bigram.item()}")
print(f"KL Divergence Transformer Bigram:{kl_transformer_bigram.item()}")

    
