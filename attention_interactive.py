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
from matplotlib.widgets import Slider
from matplotlib.collections import LineCollection
import numpy as np
import matplotlib.cm as cm

#Hyperparams
parser=argparse.ArgumentParser("Visualizing attention patterns in a trained model")
parser.add_argument("--T", type=int, default=128, help="Length of Markov chain")
parser.add_argument("--S", type=int, default=5, help="Vocab size")
parser.add_argument("--chkpt", type=str, default='checkpoint.pt', help="Path to trained weights")
parser.add_argument('--M', type=int, default=128, help='Number of past instances in RPE')
parser.add_argument('--num_heads1', type=int, default=1, help="Number of heads in layer1")
parser.add_argument('--num_heads2', type=int, default=1, help="Number of heads in layer2")
args=parser.parse_args()

T= args.T
S= args.S
chkpt=args.chkpt
M=args.M
num_heads1=args.num_heads1
num_heads2=args.num_heads2

device= "cuda" if torch.cuda.is_available() else "cpu"

model=InductionHead(S, num_heads1, num_heads2, M, S).to(device)
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
#print(attn_scores_layer1.shape)
attn_scores_layer2 = model.attn2.attention_scores.squeeze(0)
#print(attn_scores_layer2.shape)

#selected_tokens = [0,10,20,30,40,50,60,70,80,90,100,110,120,127]

num_heads = attn_scores_layer1.shape[0]

def plot_token_attention_per_head_xaxis(attn_scores_layer1, attn_scores_layer2, num_heads1, num_heads2, tokens, token_idx=127):
    T = len(tokens)
    x_labels = [str(tok) for tok in tokens]
    
    fig, axs = plt.subplots(num_heads1+num_heads2, 1, figsize=(20, 16))  # 4 plots vertically stacked
    plt.subplots_adjust(hspace=0.5)
    
    head_info = [
        (attn_scores_layer1 if j==1 else attn_scores_layer2, i, f"Layer {j} - Head {i}") for j in range(1,3) for i in range(num_heads1) 
    ]
    print(head_info)

    for ax, (attn_scores, head, title) in zip(axs, head_info):
        print(attn_scores)
        attn_scores[attn_scores<=-1e9]=0
        attention_weights = attn_scores[head, token_idx, :T].detach().cpu().numpy()

        # Normalize attention weights for color mapping
        norm = plt.Normalize(vmin=attention_weights.min(), vmax=attention_weights.max())
        colors = cm.viridis(norm(attention_weights))

        bars = ax.bar(range(T), attention_weights, color=colors, alpha=0.7)
        ax.set_title(f'{title} | Attention FROM Token {token_idx}', fontsize=16)
        ax.set_xlabel('Token Index', fontsize=14)
        ax.set_ylabel('Attention Weight', fontsize=14)
        ax.set_xticks(range(T))
        ax.set_xticklabels(x_labels, rotation=0, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)

        # Add colorbar for the attention weight color mapping
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Attention Weight', fontsize=12)

    plt.suptitle(f'Attention FROM Token {token_idx} across All Heads', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(f'attn_token{token_idx}_barplots.png', bbox_inches='tight')
    print(f"Saved attention barplots: attn_token{token_idx}_barplots.png")
    plt.close()


#plotting
plot_token_attention_per_head_xaxis(attn_scores_layer1, attn_scores_layer2, num_heads1, num_heads2, x_vals[:T], token_idx=127)


def plot_attention_heatmaps_all_heads(attn_scores_layer1, attn_scores_layer2):
    """
    Plots heatmaps of full attention matrices for all heads in layers 1 and 2.
    No axis labels or ticks to keep the heatmaps clean.
    
    Parameters:
    - attn_scores_layer1: tensor of shape [num_heads, seq_len, seq_len]
    - attn_scores_layer2: tensor of shape [num_heads, seq_len, seq_len]
    """
    num_heads1 = attn_scores_layer1.shape[0]
    num_heads2 = attn_scores_layer2.shape[0]
    
    total_plots = num_heads1 + num_heads2
    fig, axs = plt.subplots(total_plots, 1, figsize=(8, 4 * total_plots))
    plt.subplots_adjust(hspace=0.5)

    layers_info = [
        (attn_scores_layer1, num_heads1, 1),
        (attn_scores_layer2, num_heads2, 2)
    ]
    
    for plot_idx, (attn_scores, num_heads, layer_num) in enumerate(layers_info):
        for head in range(num_heads):
            idx = plot_idx * max(num_heads1, num_heads2) + head
            ax = axs[idx] if total_plots > 1 else axs
            
            attention_matrix = attn_scores[head].detach().cpu().numpy()
            sns.heatmap(attention_matrix, cmap='viridis', ax=ax, cbar=True,
                        xticklabels=False, yticklabels=False,
                        square=True, linewidths=0)
            ax.set_title(f'Layer {layer_num} - Head {head}', fontsize=14)
            ax.axis('off')  # Hide axis ticks and labels

    plt.suptitle('Attention Heatmaps for All Heads in Layers 1 and 2', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("attention_heatmap_clean.png", bbox_inches='tight')
    print("Saved attention heatmaps to 'attention_heatmap_clean.png'")
    plt.close()

plot_attention_heatmaps_all_heads(attn_scores_layer1, attn_scores_layer2)
