import torch
import torch.nn as nn 
import torchvision.models as models
import torch.nn.functional as F
import random

class CNN(nn.Module):
    def __init__(self, dimension_lambda=512):
        super(CNN, self).__init__()
        self.cnn = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
        self.cnn.fc = nn.Identity()
        self.fc_jigsaw = nn.Linear(in_features=64, out_features=dimension_lambda)
        self.activation = nn.ReLU()

    def forward(self, X):
        """
        Renvoie le vecteur de lambda_i (B, 9, dimension_lambda) à partir de X=veteur des patch mélangés (B, 9, 3, 32, 32)
        """
        batch_size, num_pieces, colors, hauteur, largeur = X.size()
        X = X.view(batch_size * num_pieces, colors, hauteur, largeur)
        X_cnn = self.cnn(X)
        lambda_flat = self.fc_jigsaw(X_cnn)
        lambda_flat = self.activation(lambda_flat)
        lambdas_i = lambda_flat.view(batch_size, num_pieces, -1)
        return lambdas_i
    

class PointerNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, attention_dim):
        super(PointerNetwork, self).__init__()
        self.encoder = nn.GRU(input_size=input_dim, 
                              hidden_size=hidden_dim,
                              num_layers=2,
                              batch_first=True,
                              bidirectional=True)
        self.decoder = nn.GRU(input_size=input_dim,
                              hidden_size=2*hidden_dim,
                              num_layers=2,
                              batch_first=True)
        self.W1 = nn.Linear(hidden_dim * 2, attention_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim * 2, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)
        self.decoder_start_input = nn.Parameter(torch.randn(1, 1, input_dim))

    def forward(self, X, teacher_forcing_ratio=0.0, target_indices=None):
        """
        Renvoie le vecteur des proba (B, 9, 9)
        """
        batch_size, num_patch, _ = X.size()
        encoder_outputs, hidden = self.encoder(X)
        hidden = hidden.view(2, 2, batch_size, -1)
        decoder_hidden = torch.cat((hidden[0], hidden[1]), dim=2)
        W1_e = self.W1(encoder_outputs)
        decoder_input = self.decoder_start_input.expand(batch_size, 1, -1)
        pointer_logits = []
        mask = torch.zeros(batch_size, num_patch, dtype=torch.bool, device=X.device)
        for t in range(num_patch):
            dec_out, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            W2_d = self.W2(dec_out)
            ui = self.v(torch.tanh(W2_d + W1_e)).squeeze(-1)
            ui_masked = ui.masked_fill(mask, -1e8)
            proba = F.softmax(ui_masked)
            best_guess = proba.argmax(dim=-1)
            pointer_logits.append(ui_masked.squeeze(-1))
            use_teacher_forcing = target_indices is not None and random.random() < teacher_forcing_ratio
            next_indices = target_indices[:, t] if use_teacher_forcing else best_guess
            nouveau_masque_etape = F.one_hot(next_indices, num_classes=num_patch).bool()
            mask = mask | nouveau_masque_etape
            decoder_input = torch.gather(X, 1, next_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, X.size(-1)))
        return torch.stack(pointer_logits, dim=1)
    
class GlobalPredictor(nn.Module):
    def __init__(self, dimension_lambda=512, hidden_dim=256, attention_dim=256, teacher_forcing_ratio=0.0, target_indices=None):
        super(GlobalPredictor, self).__init__()
        self.cnn = CNN(dimension_lambda=dimension_lambda)
        self.pointer = PointerNetwork(input_dim=dimension_lambda, 
                                      hidden_dim=hidden_dim, 
                                      attention_dim=attention_dim)
    
    def forward(self, X, teacher_forcing_ratio=0.0, target_indices=None):
        """
        Renvoie l'ordre dans lequel les patchs ont été mélangé (B, 9)
        """
        lambda_i = self.cnn(X)
        pointer_logits = self.pointer(lambda_i, teacher_forcing_ratio, target_indices)
        return pointer_logits
    
    def predict_order(self, X):
        self.eval()
        with torch.no_grad():
            logits = self.forward(X, teacher_forcing_ratio=0.0, target_indices=None)
            ordre_predit = logits.argmax(dim=-1)
        self.train()
        return ordre_predit