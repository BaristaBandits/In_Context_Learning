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
import argparse
import json
import os
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


def initialize_weights(model, init_config=None):
    def apply_init(tensor, init_type, param):
        if init_type == 'default':
            return  
        elif init_type == 'uniform':
            nn.init.uniform_(tensor, -param, param)
        elif init_type == 'normal':
            nn.init.normal_(tensor, mean=0.0, std=param)
        elif init_type == 'constant':
            nn.init.constant_(tensor, param)
        else:
            raise ValueError(f"Unknown init_type: {init_type}")

    if init_config is None:
        return  

    # Initialize Disentangled_MHSA layers
    for attn in [model.attn1, model.attn2]:
        if 'qk' in init_config:
            init_type, param = init_config['qk']
            apply_init(attn.w_q.weight, init_type, param)
            apply_init(attn.w_k.weight, init_type, param)

        if 'v' in init_config:
            init_type, param = init_config['v']
            apply_init(attn.w_v.weight, init_type, param)

    # Initialize final output projection
    if 'output_proj' in init_config:
        init_type, param = init_config['output_proj']
        apply_init(model.w_o.weight, init_type, param)
        if model.w_o.bias is not None:
            nn.init.zeros_(model.w_o.bias)


device = 'cuda' if torch.cuda.is_available() else 'cpu'


#Hyperparameters
parser=argparse.ArgumentParser(description="Train an Induction Transformer on a Markov Chain")
parser.add_argument('--T', type=int, default=128, help='Length of Markov chain : default 128')
parser.add_argument('--S', type=int, default = 5, help='Number of states in a Markov Chain: default 5')
parser.add_argument('--M', type=int, default=128, help='Number of past instances in RPE')
parser.add_argument('--num_epochs', type=int, default=1000000, help='Number of training iterations')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
parser.add_argument('--order', type=int, default=1, help='Order of markov chain')
parser.add_argument('--checkpolint', type=str, default="checkpoint.pt", help='Checkpoint to be loaded')
parser.add_argument('--load_checkpoint', action='store_true', help='Whether to load model from checkpoint')
parser.add_argument('--initialize', type=str, default=None, help="Method of initializing the weights JSON string" )
parser.add_argument('--save_chkpt', type=str, default="run1.pt", help="path to store the checkpoint")
args = parser.parse_args()

T = args.T
S = args.S
M = args.M
num_epochs = args.num_epochs
lr = args.lr
batch_size = args.batch_size
order = args.order
vocab_size = S
initialize=args.initialize
save_chkpt=args.save_chkpt

if args.initialize:
    initialize = json.loads(args.initialize)
    print(initialize)

induction_transformer=InductionHead(S,1,1,M,S).to(device)        #Initialization
criterion=nn.CrossEntropyLoss()                                  #Loss Function
optimizer=optim.Adam(induction_transformer.parameters(), lr=lr)    #Optimizer

#Initialize weights
initialize_weights(induction_transformer, initialize)
#Loading from checkpoint
if args.load_checkpoint:
    try:
        checkpoint = torch.load(args.checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        induction_transformer.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Loaded model and optimizer from {args.checkpoint_path}")
    except FileNotFoundError:
        print(f"Checkpoint file {args.checkpoint_path} not found. Training from scratch.")

induction_transformer.to(device)
induction_transformer.train()



#report to wandb
wandb.init(
    project="induction-transformer",  # name of your project
    config={
        "learning_rate": lr,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "order": order,
        "vocab_size": vocab_size,
        "sequence_length": T,
        "device": device,
        "initialize": initialize,
    }
)

mc = MarkovChain(order, vocab_size, device, t_matrix_in_context=True)
save_dir = os.path.join("runs", save_chkpt)
os.makedirs(save_dir, exist_ok=True)

#Training
for epoch in range(num_epochs):
  data = mc.get_batch(seq_length=T+1, batch_size=batch_size, initial='steady').to(device)
  src_data = data[:, :-1]  # (batch, T) - we are resitrciting it till T-1 steps
  tgt_data = data[:, 1:]   # (batch, T) — next tokens from 1 to T (left shifted)
  src_data = OneHotEmbed(S)(src_data)
  optimizer.zero_grad()
  output=induction_transformer(src_data)
  loss = criterion(
        output.contiguous().view(-1, S),      # logits
        tgt_data.contiguous().view(-1)        # true next token
    )
  loss.backward()
  optimizer.step()

  # Log to wandb
  wandb.log({"loss": loss.item(), "epoch": epoch})
  
  if epoch%100==0:
    print(f"Iteration:{epoch}. Loss:{loss.item()}")

  #print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
  # Save checkpoint every 10,000 steps
  if epoch % 10000 == 0 and epoch != 0:
        
        checkpoint_path = os.path.join(save_dir, f"checkpoint_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': induction_transformer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)
        print(f"Checkpoint saved at step {epoch} -> {checkpoint_path}")

torch.save({
    'model_state_dict': induction_transformer.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, save_chkpt)

wandb.finish()
