import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------
# 1. Original Boilerplate (Unchanged)
# ------------------------------------------------------
torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_data(n=512):
    x = torch.linspace(-np.pi, np.pi, n).unsqueeze(1)
    y = torch.sin(x)
    return x.to(device), y.to(device)

X_train, Y_train = generate_data()

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

def get_weights(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

# ------------------------------------------------------
# 2. Gradient-Only Optimization (New)
# ------------------------------------------------------
def reconstruct_weights(W0, R, N, shapes, sizes):
    """Convert flat W0 - R@N back to layer weights"""
    W = W0 - R @ N
    params = []
    start = 0
    for size, shape in zip(sizes, shapes):
        params.append(W[start:start+size].view(shape))
        start += size
    return params

def gradient_only_train(model, X, Y, latent_dim=20, n_iter=1000):
    # Initialize fixed components
    W0 = get_weights(model)
    shapes = [p.shape for p in model.parameters()]
    sizes = [p.numel() for p in model.parameters()]
    R = torch.randn(W0.numel(), latent_dim, device=device) / np.sqrt(latent_dim)
    
    # Optimization state
    N = torch.zeros(latent_dim, device=device)
    lr = 0.1
    momentum = 0.9
    velocity = torch.zeros_like(N)
    h = 1e-5  # Finite difference step
    losses = []
    
    for i in range(n_iter):
        # Current loss
        params = reconstruct_weights(W0, R, N, shapes, sizes)
        with torch.no_grad():
            x = torch.sigmoid(F.linear(X, params[0], params[1]))
            x = torch.sigmoid(F.linear(x, params[2], params[3]))
            pred = F.linear(x, params[4], params[5])
            current_loss = F.mse_loss(pred, Y).item()
        losses.append(current_loss)
        
        # Finite-difference gradient
        grad = torch.zeros_like(N)
        for j in range(latent_dim):
            N_perturbed = N.clone()
            N_perturbed[j] += h
            params = reconstruct_weights(W0, R, N_perturbed, shapes, sizes)
            with torch.no_grad():
                x = torch.sigmoid(F.linear(X, params[0], params[1]))
                x = torch.sigmoid(F.linear(x, params[2], params[3]))
                pred = F.linear(x, params[4], params[5])
                perturbed_loss = F.mse_loss(pred, Y).item()
            grad[j] = (perturbed_loss - current_loss) / h
        
        # Update with momentum
        velocity = momentum * velocity + (1 - momentum) * grad
        N -= lr * velocity
        
        if i % 30 == 0:
            print(f"Iter {i:4d} | Loss: {current_loss:.5f}")
    
    return N, R, losses

# ------------------------------------------------------
# 3. Training & Visualization (Modified)
# ------------------------------------------------------
model = MLP().to(device)
N_optimal, R_optimal, losses = gradient_only_train(model, X_train, Y_train)

# Reconstruct final model
with torch.no_grad():
    W_final = get_weights(model) - R_optimal @ N_optimal
    start = 0
    model_final = MLP().to(device)
    for p in model_final.parameters():
        size = p.numel()
        p.copy_(W_final[start:start+size].view_as(p))
        start += size

# Plot results
with torch.no_grad():
    pred = model_final(X_train)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(X_train.cpu(), Y_train.cpu(), label='Target')
plt.plot(X_train.cpu(), pred.cpu(), label='Predicted')
plt.legend()
plt.title(f"Final Prediction (Loss: {losses[-1]:.4f})")

plt.subplot(1,2,2)
plt.plot(losses)
plt.title("Optimization Progress")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.tight_layout()
plt.show()
