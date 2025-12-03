from sklearn.metrics import r2_score
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
import lightgbm as lgb  
import warnings
warnings.filterwarnings("ignore")

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
    alpha=1.0, beta=1.0,
    optimizer_cls=torch.optim.Adam,
    lr=1e-3,
    loss_rec_cls=nn.MSELoss,  # nn.MSELoss ou nn.SmoothL1Loss
    loss_sup_cls=nn.MSELoss,
    weight_decay=0.0
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Move data to GPU
    X_train_t = X_train_t.to(device)
    y_train_t = y_train_t.to(device)
    X_test_t  = X_test_t.to(device)
    y_test_t  = y_test_t.to(device)

    # Prepare numpy data once
    y_test_np = y_test_t.cpu().numpy()
    y_train_np = y_train_t.cpu().numpy()
    
    n = len(X_train_t)
    
    # Initialize Optimizer and Loss Functions (Instances)
    optimizer = optimizer_cls(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay
    )
    criterion_rec = loss_rec_cls().to(device)
    criterion_sup = loss_sup_cls().to(device)
    
    scaler = GradScaler()

    r2_rec_test_list = []
    r2_sup_test_list = []
    r2_lgbm_test_list = []
    loss_list = []

    for epoch in range(n_epochs):
        model.train() 
        perm = torch.randperm(n, device=device)
        total_loss = 0.0

        # --- TRAIN LOOP ---
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb = X_train_t[idx]
            yb = y_train_t[idx]

            optimizer.zero_grad()

            with autocast():
                z, x_hat, y_hat = model(xb)

                loss_rec = criterion_rec(x_hat, xb)
                loss_sup = criterion_sup(y_hat, yb)
                loss = alpha * loss_rec + beta * loss_sup

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        # --- EVALUATION ---
        model.eval()
        with torch.no_grad():
            z_train, _, _ = model(X_train_t)
            z_test, x_hat_test, y_hat_test = model(X_test_t)
            
        # Convert embeddings for LGBM on CPU
        z_train_np = z_train.cpu().numpy()
        z_test_np = z_test.cpu().numpy()
        
        # Train LGBM (on Z_train)
        lgbm_model = lgb.LGBMRegressor(objective='regression', metric='rmse', 
                                       n_estimators=100, verbose=-1, n_jobs=1)
        lgbm_model.fit(z_train_np, y_train_np.flatten())

        # Evaluate LGBM (on Z_test)
        y_lgbm_pred = lgbm_model.predict(z_test_np)
        r2_lgbm_test = r2_score(y_test_np.flatten(), y_lgbm_pred)
        r2_lgbm_test_list.append(r2_lgbm_test)

        # Calculate R2 for NN TaskHead
        r2_rec_test = r2_score(X_test_t.cpu().numpy(), x_hat_test.cpu().numpy())
        r2_sup_test = r2_score(y_test_t.cpu().numpy(), y_hat_test.cpu().numpy())

        r2_rec_test_list.append(r2_rec_test)
        r2_sup_test_list.append(r2_sup_test)
        loss_list.append(total_loss)

        print(
            f"Epoch {epoch+1:02d} | loss={total_loss:.3f} "
            f"| R2_sup_NN={r2_sup_test:.4f} "
            f"| R2_sup_LGBM={r2_lgbm_test:.4f} "
            f"| R2_rec={r2_rec_test:.4f}"
        )

    return r2_rec_test_list, r2_sup_test_list, r2_lgbm_test_list, loss_list

def plot_r2(r2_rec_list, r2_sup_list, r2_lgbm_list):
    """
    Plots the evolution of three R^2 metrics over epochs:
    1. Reconstruction (Autoencoder)
    2. Supervision (TaskHead)
    3. LightGBM (LGBM model trained on the latent code Z)
    """
    # Create 3 subplots stacked vertically, sharing the X-axis (Epoch)
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # --- subplot 1 : reconstruction (Reconstruction R²) ---
    axes[0].plot(r2_rec_list, color="tab:blue")
    axes[0].set_ylabel("R² (Reconstruction)")
    axes[0].set_title("Evolution of R² Reconstruction (Encoder/Decoder)")
    axes[0].grid(True)

    # --- subplot 2 : supervision (NN TaskHead R²) ---
    axes[1].plot(r2_sup_list, color="tab:green")
    axes[1].set_ylabel("R² (NN Supervision)")
    axes[1].set_title("Evolution of R² Supervision (NN TaskHead)")
    axes[1].grid(True)

    # --- subplot 3 : LightGBM (LGBM R²) ---
    axes[2].plot(r2_lgbm_list, color="tab:orange")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("R² (LGBM on Z)")
    axes[2].set_title("Evolution of R² (LightGBM on Latent Code Z)")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()