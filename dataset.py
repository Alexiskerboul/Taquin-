import torch 
import torchvision 
import torchvision.transforms as transform 
from torch.utils.data import Dataset, DataLoader
import numpy as np 


class DatasetPuzzle(Dataset):
    """
    Cette classe a pour but de créer le dataset X = patches_mélangés et Y = ordre_de_mélange
    """
    def __init__(self, train=True):
        self.split_name = 'train' if train else 'test'

        self.data = torchvision.datasets.STL10(root='./data',
                                       split=self.split_name,
                                       download=True,
                                       transform=transform.Compose([
                                           transform.ToTensor(), 
                                           transform.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713))
                                       ]))
        
    def __len__(self):
        """
        Renvoie la taille du dataset
        """
        return len(self.data)
    
    def __getitem__(self, indice):
        """
        Renvoie X, y, et l'ordre depuis y pour retrouver X
        """
        image, _ = self.data[indice]
        patches = []
        for i in range(3):
            for j in range(3):
                patch = image[:, i*32:(i+1)*32, j*32:(j+1)*32]
                patches.append(patch)
        patches = torch.stack(patches)
        permutation = torch.randperm(9)
        patches_melange = patches[permutation]
        return patches, patches_melange, permutation