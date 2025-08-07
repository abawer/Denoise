import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------
# 1. Synthetic Dataset: sin(x)
# ------------------------------------------------------
torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_data(n=512):
    x = torch.linspace(-np.pi, np.pi, n).unsqueeze(1)
    y = torch.sin(x)
    return x.to(device), y.to(device)

X_train, Y_train = generate_data()

# ------------------------------------------------------
# 2. Simple MLP definition with Xavier init
# ------------------------------------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)
        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return self.fc3(x)

# ------------------------------------------------------
# 3. Flatten & inject weights
# ------------------------------------------------------
def get_weights(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

# ------------------------------------------------------
# 4. Loss Function
# ------------------------------------------------------
def compute_loss(pred, y):
    return F.mse_loss(pred, y)

# ------------------------------------------------------
# 5. Initialization
# ------------------------------------------------------
model = MLP().to(device)
W = get_weights(model)
W_dim = W.numel()
latent_dim = 20

print(W_dim, latent_dim)

R = (torch.randn(W_dim, latent_dim, device=device) / latent_dim**0.5)
N = nn.Parameter(torch.zeros(latent_dim, device=device))  # Make N a learnable parameter

# ------------------------------------------------------
# 6. Latent Optimization Loop with Backpropagation (FIXED)
# ------------------------------------------------------
lr_N = 0.01
n_iter = 5000
losses = []

optimizer = torch.optim.Adam([N], lr=lr_N)

# Extract parameter shapes for reconstruction
shapes = [p.data.shape for p in model.parameters()]
sizes = [p.numel() for p in model.parameters()]

for i in range(n_iter + 1):
    # Compute current weights
    W_current = W - R @ N
    
    # Reconstruct parameters from flat vector
    params = []
    start = 0
    for size, shape in zip(sizes, shapes):
        param = W_current[start:start+size].view(shape)
        params.append(param)
        start += size
    
    # Functional forward pass (preserves gradient flow)
    x = X_train
    x = torch.sigmoid(F.linear(x, params[0], params[1]))
    x = torch.sigmoid(F.linear(x, params[2], params[3]))
    pred = F.linear(x, params[4], params[5])
    
    # Compute loss
    loss = compute_loss(pred, Y_train)
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i % 100 == 0:
        print(f"Iter {i:4d} | Loss: {loss.item():.5f} | ||N||: {N.data.norm().item():.4f}")
    
    losses.append(loss.item())

# ------------------------------------------------------
# 7. Visualize Final Result
# ------------------------------------------------------
# Recreate final model with learned weights
model_final = MLP().to(device)
with torch.no_grad():
    W_final = W - R @ N
    start = 0
    for i, p in enumerate(model_final.parameters()):
        size = p.numel()
        p.copy_(W_final[start:start+size].view_as(p))
        start += size

with torch.no_grad():
    pred = model_final(X_train)

plt.figure(figsize=(8, 4))
plt.plot(X_train.cpu(), Y_train.cpu(), label="Target", linewidth=2)
plt.plot(X_train.cpu(), pred.cpu(), label="Predicted", linewidth=2)
plt.legend()
plt.title("Final Model after Latent Optimization (Backprop)")
plt.grid()
plt.show()

plt.figure()
plt.plot(losses)
plt.title("Loss (Backprop) over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid()
plt.show()
