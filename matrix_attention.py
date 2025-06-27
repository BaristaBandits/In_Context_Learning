#This script helps visualize attentions

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
import seaborn as sns
import os 
import numpy as np

def save_tensor_as_txt(tensor, filename):
    np_array = tensor.cpu().detach().numpy()
    np.savetxt(filename, np_array, fmt='%.6f')

# Create directory
save_dir = "saved_matrices_txt"
os.makedirs(save_dir, exist_ok=True)

#Hyperparams
parser=argparse.ArgumentParser("Visualizing attention patterns in a trained model")
parser.add_argument("--T", type=int, default=128, help="Length of Markov chain")
parser.add_argument("--S", type=int, default=5, help="Vocab size")
parser.add_argument("--chkpt", type=str, default='checkpoint.pt', help="Path to trained weights")
parser.add_argument('--M', type=int, default=128, help='Number of past instances in RPE')
parser.add_argument('--num_heads1', type=int, default=1, help="Number of heads in layer 1")
parser.add_argument('--num_heads2', type=int, default=1, help="Number of heads in layer 2")
args=parser.parse_args()

T= args.T
S= args.S
chkpt=args.chkpt
M=args.M
num_heads1=args.num_heads1
num_heads2=args.num_heads2

device= "cuda" if torch.cuda.is_available() else "cpu"

model=InductionHead(S, num_heads1, num_heads2 , M, S).to(device)
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

def show_matrix(matrix, title):
    matrix[matrix==-1e9]=0
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix.detach().cpu(), cmap='viridis')
    plt.title(title)
    plt.xlabel("Input Dimension")
    plt.ylabel("Output Dimension")
    plt.tight_layout()
    plt.savefig('output_matrix.png')

def calculate_sparsity(weight_matrix):
    return (np.abs(weight_matrix.detach().cpu())<5e-2).sum()/weight_matrix.numel()
# -- Attention Weights -- 

# --- Layer 1 ---x
print("Layer 1:")
print(f"W_q - Sparsity = {calculate_sparsity(model.attn1.w_q.weight)}\n:", model.attn1.w_q.weight.shape)
print(model.attn1.w_q.weight)
save_tensor_as_txt(model.attn1.w_q.weight, os.path.join(save_dir, "layer1_w_q.txt"))

print(f"W_k - Sparsity = {calculate_sparsity(model.attn1.w_k.weight)}\n:", model.attn1.w_k.weight.shape)
print(model.attn1.w_k.weight)
save_tensor_as_txt(model.attn1.w_k.weight, os.path.join(save_dir, "layer1_w_k.txt"))

print(f"W_v - Sparsity = {calculate_sparsity(model.attn1.w_v.weight)}\n:", model.attn1.w_v.weight.shape)
print(model.attn1.w_v.weight)
save_tensor_as_txt(model.attn1.w_v.weight, os.path.join(save_dir, "layer1_w_v.txt"))

attn=model.attn1.w_q.weight.transpose(0, 1)@model.attn1.w_k.weight
print(attn)

# --- Layer 2 ---
print("\nLayer 2:")
print(f"W_q - Sparsity = {calculate_sparsity(model.attn2.w_q.weight)}\n:", model.attn2.w_q.weight.shape)
print(model.attn2.w_q.weight)
save_tensor_as_txt(model.attn2.w_q.weight, os.path.join(save_dir, "layer2_w_q.txt"))
print(f"W_k - Sparsity = {calculate_sparsity(model.attn2.w_k.weight)}\n:", model.attn2.w_k.weight.shape)
print(model.attn2.w_k.weight)
save_tensor_as_txt(model.attn2.w_k.weight, os.path.join(save_dir, "layer2_w_k.txt"))

print(f"W_v - Sparsity = {calculate_sparsity(model.attn2.w_v.weight)} \n:", model.attn2.w_v.weight.shape)
print(model.attn2.w_v.weight)
save_tensor_as_txt(model.attn2.w_v.weight, os.path.join(save_dir, "layer2_w_v.txt"))


# --- Final Output Projection ---
print("\nFinal output projection layer:")
print(f"W_o - Sparsity = {calculate_sparsity(model.w_o.weight)}\n:", model.w_o.weight.shape)
print(model.w_o.weight)
#show_matrix(model.w_o.weight, "Final Output Projection - W_o")
save_tensor_as_txt(model.w_o.weight, os.path.join(save_dir, "final_w_o.txt"))

# --- RPE ---
#For Layer 1
rpe1 = model.attn1.RPE(T)  
print(f"RPE for Layer 1 (attn1): Sparsity = {calculate_sparsity(rpe1)}")
print(rpe1.squeeze(-1).cpu().detach().numpy())  
np.savetxt(os.path.join(save_dir, "rpe_layer1.txt"), rpe1.squeeze(-1).cpu().detach().numpy(), fmt='%.6f')

# For Layer 2
rpe2 = model.attn2.RPE(T)
print(f"\nRPE for Layer 2 (attn2): Sparsity = {calculate_sparsity(rpe2)}")
print(rpe2.squeeze(-1).cpu().detach().numpy()) 
np.savetxt(os.path.join(save_dir, "rpe_layer2.txt"), rpe2.squeeze(-1).cpu().detach().numpy(), fmt='%.6f')

#attn=model.attn2.w_q.weight.transpose(0, 1)@model.attn2.w_k.weight
#print(attn)
show_matrix(rpe1.squeeze(-1), "rpe")