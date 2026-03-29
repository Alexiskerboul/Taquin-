import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision 
import torchvision.transforms as transform 
from torch.utils.data import Dataset, DataLoader
import numpy as np 
from architecture import GlobalPredictor
from dataset import DatasetPuzzle
from tqdm import tqdm

def teacher_forcing(epoch, n_epoch):
    ratio = 1.0 - (epoch / n_epoch)
    return max(0.0, ratio)

BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Lancement de l'entraînement sur : {device}")

trainset = DatasetPuzzle()
testset = DatasetPuzzle(train=False)
train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(testset, batch_size=BATCH_SIZE)

model = GlobalPredictor().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
n_epoch = 10
teacher_forcing_ratio = 0.5


def fit(model, criterion, optimizer, n_epoch, train_loader, test_loader, device):
    """
    Entraine le modèle et renvoie les loss et la precision
    """
    train_loss = []
    val_loss = []
    precision = []

    for epoch in range(n_epoch):
        model.train()
        loss_t = 0.0
        barre_progression = tqdm(train_loader, desc=f"Époque [{epoch+1}/{n_epoch}]", leave=False)
        for batch_idx, (patch_i, patch_m, target) in enumerate(barre_progression):
            optimizer.zero_grad()
            patch_m, target = patch_m.to(device), target.to(device)
            scores_bruts = model(patch_m, teacher_forcing_ratio=teacher_forcing(epoch, n_epoch), target_indices=target)
            loss = criterion(scores_bruts.view(-1, 9), target.view(-1))
            loss.backward()
            optimizer.step()
            loss_t += loss.item()
            barre_progression.set_postfix({'Loss': f"{loss.item():.4f}"})

        model.eval()
        loss_v = 0.0
        correct_pieces = 0
        pieces = 0
        barre_validation = tqdm(test_loader, desc=f"Validation [{epoch+1}/{n_epoch}]", leave=False)
        with torch.no_grad():
            for batch_idx, (patch_i, patch_m, target) in enumerate(barre_validation):
                patch_m, target = patch_m.to(device), target.to(device)
                scores_val = model(patch_m, teacher_forcing_ratio=0.0)
                loss = criterion(scores_val.view(-1, 9), target.view(-1))
                loss_v += loss.item()
                predicted_order = scores_val.argmax(dim=-1)
                correct_pieces += (predicted_order == target).sum().item()
                pieces += target.numel()
                barre_validation.set_postfix({'Val Loss': f"{loss.item():.4f}"})

        loss_train = loss_t / len(train_loader)
        train_loss.append(loss_train)
        loss_val = loss_v / len(test_loader)
        val_loss.append(loss_val)
        precision_val = (correct_pieces / pieces) * 100
        precision.append(precision_val)
        print(f"Époque [{epoch+1}/{n_epoch}]")
        print(f"   Train Loss : {loss_train:.4f}")
        print(f"   Val Loss   : {loss_val:.4f} | Précision : {precision_val:.2f}% des pièces bien placées")
    return train_loss, val_loss, precision