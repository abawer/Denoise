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
    y = torch.cos(x)
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
# 2. Weight reconstruction helper (unchanged)
# ------------------------------------------------------
def reconstruct_weights(W0, R, N, shapes, sizes):
    W = W0 - R @ N
    params = []
    start = 0
    for size, shape in zip(sizes, shapes):
        params.append(W[start:start+size].view(shape))
        start += size
    return params

# ------------------------------------------------------
# 3. Model forward with given parameters (unchanged)
# ------------------------------------------------------
def model_forward(X, params):
    x = torch.sigmoid(F.linear(X, params[0], params[1]))
    x = torch.sigmoid(F.linear(x, params[2], params[3]))
    pred = F.linear(x, params[4], params[5])
    return pred

# ------------------------------------------------------
# 4. Random Direction Gradient Estimation optimizer
# ------------------------------------------------------
def latent_coordinate_descent(model, X, Y, latent_dim=32, n_iter=5000, step=1e-5, lr=0.1, momentum=0.95):
    W0 = get_weights(model)
    shapes = [p.shape for p in model.parameters()]
    sizes = [p.numel() for p in model.parameters()]
    d = W0.numel()

    R = torch.randn(d, latent_dim, device=W0.device) * latent_dim ** -0.5
    N = torch.zeros(latent_dim, device=W0.device)
    velocity = torch.zeros_like(N)  # momentum accumulator

    losses = []

    k = int(latent_dim**0.5)  # number of random directions to average over
    h = step

    for i in range(n_iter + 1):
        grad_estimate = torch.zeros_like(N)

        for _ in range(k):
            u = torch.randn(latent_dim, device=N.device)
            u /= torch.norm(u) + 1e-8  # normalize

            N_plus = N + h * u
            N_minus = N - h * u

            params_plus = reconstruct_weights(W0, R, N_plus, shapes, sizes)
            params_minus = reconstruct_weights(W0, R, N_minus, shapes, sizes)

            with torch.no_grad():
                loss_plus = F.mse_loss(model_forward(X, params_plus), Y).item()
                loss_minus = F.mse_loss(model_forward(X, params_minus), Y).item()

            grad_estimate += ((loss_plus - loss_minus) / (2 * h)) * u

        grad_estimate /= k  # average

        # Momentum update
        velocity = momentum * velocity - lr * grad_estimate
        N = N + velocity

        # Track loss at current N
        params = reconstruct_weights(W0, R, N, shapes, sizes)
        with torch.no_grad():
            pred = model_forward(X, params)
            loss_now = F.mse_loss(pred, Y).item()
        losses.append(loss_now)

        if i % 100 == 0:
            print(f"Iter {i:4d} | Loss: {loss_now:.5f}")

    return N, R, losses


# ------------------------------------------------------
# 5. Train and visualize
# ------------------------------------------------------
model = MLP().to(device)
N_optimal, R_optimal, losses = latent_coordinate_descent(model, X_train, Y_train)

# Final model reconstruction
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
