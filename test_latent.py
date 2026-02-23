import torch
import torch.nn as nn
import torch.nn.functional as F

X_t = torch.randn(135, 1, 18)
latent_dim = 64

encoder = nn.Sequential(
    nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm1d(16),
    nn.ReLU(),
    nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
    nn.BatchNorm1d(32),
    nn.ReLU(),
    nn.AdaptiveAvgPool1d(latent_dim // 32),
    nn.Flatten()
)

dummy_out = encoder(X_t)
flat_size = dummy_out.shape[1]

decoder = nn.Sequential(
    nn.Linear(flat_size, 32 * (X_t.shape[2] // 4)),
    nn.Unflatten(1, (32, X_t.shape[2] // 4)),
    nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
    nn.BatchNorm1d(16),
    nn.ReLU(),
    nn.ConvTranspose1d(16, 1, kernel_size=7, stride=2, padding=3, output_padding=1)
)

Z = encoder(X_t)
X_rec = decoder(Z)
if X_rec.shape[2] != X_t.shape[2]:
    X_rec = F.interpolate(X_rec, size=X_t.shape[2])
loss = F.mse_loss(X_rec, X_t)
loss.backward()
print("Success")
