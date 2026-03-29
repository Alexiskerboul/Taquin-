import numpy as np 
import torch

def creation_image_numpy(patch, h_del):
    """
    Renvoie la l'image avec un patch vide pour h_del
    """
    patch = patch.squeeze(0)
    image = torch.zeros(3, 96, 96)
    for h in range(9):
        if h != h_del:
            j = h % 3
            i = h // 3
            image[:, i*32:(i+1)*32, j*32:(j+1)*32] = patch[h]
    image = image.numpy()
    image = np.transpose(image, (1, 2, 0))
    image = np.clip(image, 0, 1)
    return image

def creation_image_pg(image):
    """
    Renvoie une image prête pour pygame
    """
    image = (image * 255).astype(np.uint8)
    image_pygame = np.transpose(image, (1, 0, 2))
    return image_pygame

def change_place(image, i_del, j_del, i_cible, j_cible):
    """
    Change la place de deux patchs 
    """
    temp = image[i_cible*128:(i_cible+1)*128, j_cible*128:(j_cible+1)*128].copy()
    image[i_cible*128:(i_cible+1)*128, j_cible*128:(j_cible+1)*128] = image[i_del*128:(i_del+1)*128, j_del*128:(j_del+1)*128]
    image[i_del*128:(i_del+1)*128, j_del*128:(j_del+1)*128] = temp
    return image
