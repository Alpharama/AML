import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
import warnings
warnings.filterwarnings("ignore")

def make_mlp(widths, activation=nn.ReLU):
    """
    widths : [in_dim, h1, h2, ..., out_dim]
    """
    layers = []
    for in_f, out_f in zip(widths[:-1], widths[1:]):
        layers.append(nn.Linear(in_f, out_f))
        layers.append(activation())
    layers.pop()  # retire la dernière activation inutile
    return nn.Sequential(*layers)

class Encoder(nn.Module):
    def __init__(self, widths):
        super().__init__()
        self.net = make_mlp(widths)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, widths):
        super().__init__()
        self.net = make_mlp(widths)

    def forward(self, z):
        return self.net(z)


class TaskHead(nn.Module):
    def __init__(self, widths):
        super().__init__()
        self.net = make_mlp(widths)

    def forward(self, z):
        return self.net(z)

class FullModel(nn.Module):
    def __init__(self,
                 encoder_widths,   # ex : [input_dim, 256, 128, latent_dim]
                 decoder_widths,   # ex : [latent_dim, 128, 256, input_dim]
                 head_widths       # ex : [latent_dim, 64, output_dim]
                ):
        super().__init__()

        self.encoder = Encoder(encoder_widths)
        self.decoder = Decoder(decoder_widths)
        self.head = TaskHead(head_widths)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        y_hat = self.head(z)
        return z, x_hat, y_hat

def r2_loss(y_true, y_pred):
    """
    R^2 est calculé via sklearn → numpy nécessaire.
    Loss = 1 - R^2 pour pouvoir backpropager.
    """
    y_t = y_true.detach().cpu().numpy()
    y_p = y_pred.detach().cpu().numpy()
    return 1 - r2_score(y_t, y_p)


def train_model(
    model,
    X_train_t, y_train_t,
    X_test_t, y_test_t,
    n_epochs=60, batch_size=64,
    alpha=1.0, beta=1.0
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # envoyer données sur GPU
    X_train_t = X_train_t.to(device)
    y_train_t = y_train_t.to(device)
    X_test_t  = X_test_t.to(device)
    y_test_t  = y_test_t.to(device)

    n = len(X_train_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler()

    r2_rec_test_list = []
    r2_sup_test_list = []
    loss_list = []

    for epoch in range(n_epochs):

        perm = torch.randperm(n, device=device)
        total_loss = 0.0

        # -------------------------
        #   TRAIN LOOP
        # -------------------------
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb = X_train_t[idx]
            yb = y_train_t[idx]

            optimizer.zero_grad()

            with autocast():
                z, x_hat, y_hat = model(xb)

                loss_rec = F.mse_loss(x_hat, xb)
                loss_sup = F.mse_loss(y_hat, yb)
                loss = alpha * loss_rec + beta * loss_sup

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        # -------------------------
        #   R² SUR LE TEST SET
        # -------------------------
        with torch.no_grad():
            _, x_hat_test, y_hat_test = model(X_test_t)

        # convertir vers CPU + numpy
        r2_rec_test = r2_score(X_test_t.cpu().numpy(),
                               x_hat_test.cpu().numpy())
        r2_sup_test = r2_score(y_test_t.cpu().numpy(),
                               y_hat_test.cpu().numpy())

        r2_rec_test_list.append(r2_rec_test)
        r2_sup_test_list.append(r2_sup_test)
        loss_list.append(total_loss)

        print(
            f"Epoch {epoch+1:02d} | loss={total_loss:.3f} "
            f"| R2_rec_test={r2_rec_test:.4f} "
            f"| R2_sup_test={r2_sup_test:.4f}"
        )

    return r2_rec_test_list, r2_sup_test_list, loss_list


def plot_r2(r2_rec_list, r2_sup_list):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- subplot 1 : reconstruction ---
    axes[0].plot(r2_rec_list, color="tab:blue")
    axes[0].set_ylabel("R² (reconstruction)")
    axes[0].set_title("Évolution du R² reconstruction")
    axes[0].grid(True)

    # --- subplot 2 : supervision ---a
    axes[1].plot(r2_sup_list, color="tab:green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("R² (supervision)")
    axes[1].set_title("Évolution du R² supervision")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()