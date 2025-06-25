#This script helps visualize attentions

import seaborn as sns 
from model import InductionHead, OneHotEmbed
import argparse
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

#Hyperparams
parser=argparse.ArgumentParser("Visualizing attention patterns in a trained model")
parser.add_argument("--T", type=int, default=128, help="Length of Markov chain")
parser.add_argument("--S", type=int, default=5, help="Vocab size")
parser.add_argument("--chkpt", type=str, default='checkpoint.pt', help="Path to trained weights")
parser.add_argument('--M', type=int, default=128, help='Number of past instances in RPE')
args=parser.parse_args()

T= args.T
S= args.S
chkpt=args.chkpt
M=args.M

device= "cuda" if torch.cuda.is_available() else "cpu"

model=InductionHead(S, 2, 2, M, S).to(device)
checkpoint= torch.load(chkpt, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

#Sampling a Markov Test chain for testing
alpha=torch.tensor([1.0]*S, device=device)
transition_matrix=torch.distributions.Dirichlet(alpha).sample((S,))
#print(transition_matrix.shape)
initial_distribution= torch. full((S, ), 1.0/S, device=device)

current_state=torch.multinomial(initial_distribution, 1).item()
chain=[current_state]

for _ in range(T-1):
    probs = transition_matrix[current_state]
    next_state=torch.multinomial(probs, 1).item()
    chain.append(next_state)
    current_state=next_state

mc=torch.tensor(chain, device=device)
#print(mc.shape)
x_vals = mc.tolist()

#Forward Pass
input = OneHotEmbed(S)(mc.unsqueeze(0))
output=model(input)

attn_scores_layer1 = model.attn1.attention_scores.squeeze(0) # shape: ( num_heads, seq_len, seq_len)
print(attn_scores_layer1.shape)
attn_scores_layer2 = model.attn2.attention_scores.squeeze(0)
print(attn_scores_layer2.shape)

selected_tokens = [0,10,20,30,40,50,60,70,80,90,100,110,120,127]

num_heads = attn_scores_layer1.shape[0]

for token_idx in selected_tokens:
    fig, axs = plt.subplots(2, num_heads, figsize=(4 * num_heads, 6))
    
    for head in range(num_heads):
        # Layer 1 heatmap
        sns.heatmap(
            attn_scores_layer1[head, token_idx].unsqueeze(0).cpu().detach().numpy(),
            ax=axs[0, head], cmap='viridis', cbar=False,
            xticklabels=x_vals, yticklabels=[]
        )
        axs[0, head].set_title(f'Layer 1 - Head {head} - Token {token_idx}')
        axs[0, head].set_xticks(range(len(x_vals)))
        axs[0, head].tick_params(axis='x', rotation=90)
        
        # Layer 2 heatmap
        sns.heatmap(
            attn_scores_layer2[head, token_idx].unsqueeze(0).cpu().detach().numpy(),
            ax=axs[1, head], cmap='viridis', cbar=False,
            xticklabels=x_vals, yticklabels=[]
        )
        axs[1, head].set_title(f'Layer 2 - Head {head} - Token {token_idx}')
        axs[1, head].set_xticks(range(len(x_vals)))
        axs[1, head].tick_params(axis='x', rotation=90)
    
    plt.tight_layout()
    plt.savefig(f'attention_token_{token_idx}.png')
    plt.close(fig)
    