import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)

# ------------------------------------------------------------------
# 1.  Data (same as before)
# ------------------------------------------------------------------
n, d_in, d_hidden, d_out = 1000, 10, 64, 1
X = torch.randn(n, d_in, device=device)
true_mlp = torch.nn.Sequential(
    torch.nn.Linear(d_in, d_hidden),
    torch.nn.Tanh(),
    torch.nn.Linear(d_hidden, d_out)
).to(device)
with torch.no_grad():
    Y = true_mlp(X)
loader = DataLoader(TensorDataset(X, Y), batch_size=256, shuffle=True)

# ------------------------------------------------------------------
# 2.  True / noisy weights
# ------------------------------------------------------------------
true_params = [p.detach().clone() for p in true_mlp.parameters()]
noise_std = 0.5
noisy_params = [p + torch.randn_like(p) * noise_std for p in true_params]

# ------------------------------------------------------------------
# 3.  Low-rank linear noise model
# ------------------------------------------------------------------
latent_dim = 200
proj_mats, z_vecs = [], []
for p in noisy_params:
    flat = p.numel()
    # fixed random basis
    R = torch.randn(flat, latent_dim, device=device) / (latent_dim ** 0.5)
    proj_mats.append(R)
    # learnable latent vector
    z = torch.zeros(latent_dim, 1, device=device, requires_grad=True)
    z_vecs.append(z)

# ------------------------------------------------------------------
# 4.  Forward helper
# ------------------------------------------------------------------
def forward(x):
    params = [p - (R @ z).view_as(p)
              for p, R, z in zip(noisy_params, proj_mats, z_vecs)]
    w1, b1, w2, b2 = params
    h = torch.tanh(F.linear(x, w1, b1))
    return F.linear(h, w2, b2)

# ------------------------------------------------------------------
# 5.  Optimise latent vectors (linear in z, non-linear network)
# ------------------------------------------------------------------
opt = torch.optim.Adam(z_vecs, lr=1e-2)

for epoch in range(3000):
    for xb, yb in loader:
        opt.zero_grad()
        loss = F.mse_loss(forward(xb), yb)
        loss.backward()
        opt.step()
    if epoch % 50 == 0:
        with torch.no_grad():
            full_loss = F.mse_loss(forward(X), Y).item()
        print(f'Epoch {epoch:3d}  loss={full_loss:.6f}')

###############################################################################
# 6.  Evaluation & plots  (clean)
###############################################################################
# 6-a  Save the learned latent vectors
learned_z = [z.clone() for z in z_vecs]

# 6-b  Create the noisy model (z = 0)
with torch.no_grad():
    for z in z_vecs:
        z.zero_()
    y_noisy = forward(X)

# 6-c  Restore learned z for the denoised model
with torch.no_grad():
    for z, lz in zip(z_vecs, learned_z):
        z.copy_(lz)
    y_clean = forward(X)

# 6-d  Report MSEs
print(f'Noisy  MSE:   {F.mse_loss(y_noisy, Y).item():.5f}')
print(f'Denoised MSE: {F.mse_loss(y_clean, Y).item():.5f}')

# 6-e  Plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0].cpu(), Y.squeeze().cpu(), s=10)
plt.title('True Data')

plt.subplot(1, 3, 2)
plt.scatter(X[:, 0].cpu(), y_noisy.squeeze().cpu(), s=10, color='r')
plt.title('Noisy Model (z = 0)')

plt.subplot(1, 3, 3)
plt.scatter(X[:, 0].cpu(), y_clean.squeeze().cpu(), s=10, color='g')
plt.title('Denoised Model (learned z)')

plt.tight_layout()
plt.show()
