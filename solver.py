import torch
import numpy as np 
from dataset import DatasetPuzzle
from utils import creation_image_numpy, creation_image_pg, change_place
import random

trainset = DatasetPuzzle()
patch, patch_melange, permutation = trainset[300]
h_del = random.randint(0, 8)
cible = [0, 1, 2, 3, 4, 5, 6, 7, 8]

