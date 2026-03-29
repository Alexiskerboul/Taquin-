import pygame as pg 
from dataset import DatasetPuzzle
from torch.utils.data import DataLoader
from utils import creation_image_numpy, creation_image_pg, change_place
import random
import numpy as np 

def main(image, h_del, image_cible):
    """
    Lance le jeu
    """
    j_del = h_del // 3
    i_del = h_del % 3
    pg.init()
    surface_taquin = pg.surfarray.make_surface(image)
    taille_fenetre = (384, 384)
    surface_agrandie = pg.transform.scale(surface_taquin, taille_fenetre)
    ecran = pg.display.set_mode(taille_fenetre)
    pg.display.set_caption("Jeu du Taquin - STL10")

    surface_cible_petite = pg.surfarray.make_surface(image_cible)
    surface_cible = pg.transform.scale(surface_cible_petite, taille_fenetre)
    pixels_cibles = pg.surfarray.array3d(surface_cible)

    police_compteur = pg.font.SysFont(None, 36)
    compteur = 0
    en_cours = True
    partie_gagnee = False
    while en_cours:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                en_cours = False
            elif event.type == pg.KEYDOWN and not partie_gagnee:
                i_cible, j_cible = i_del, j_del
                if event.key == pg.K_UP:
                    j_cible = j_del - 1
                elif event.key == pg.K_DOWN:
                    j_cible = j_del + 1
                elif event.key == pg.K_LEFT:
                    i_cible = i_del - 1
                elif event.key == pg.K_RIGHT:
                    i_cible = i_del + 1
                if 0 <= i_cible < 3 and 0 <= j_cible < 3:
                    pixels = pg.surfarray.pixels3d(surface_agrandie)
                    change_place(pixels, i_del, j_del, i_cible, j_cible)
                    i_del, j_del = i_cible, j_cible
                    compteur += 1
                    if np.array_equal(pixels, pixels_cibles):
                        partie_gagnee = True
                    del pixels
        ecran.blit(surface_agrandie, (0, 0))
        texte_compteur = police_compteur.render(f" Coups : {compteur} ", True, (255, 255, 255), (0, 0, 0))
        ecran.blit(texte_compteur, (10, 10))
        if partie_gagnee:
            fond_sombre = pg.Surface(taille_fenetre)
            fond_sombre.set_alpha(150) 
            fond_sombre.fill((0, 0, 0))
            ecran.blit(fond_sombre, (0, 0))
            police = pg.font.SysFont(None, 64)
            texte_victoire = police.render(f"Félicitations ! Tu as gagné en {compteur} coups !", True, (50, 200, 50))
            rect_texte = texte_victoire.get_rect(center=(384 // 2, 384 // 2))
            ecran.blit(texte_victoire, rect_texte)
        pg.display.flip()

    pg.quit()


trainset = DatasetPuzzle()
patch, patch_melange, permutation, h_del = trainset[300]
image_melange = creation_image_numpy(patch_melange, h_del)
image_melange = creation_image_pg(image_melange)
image_cible = creation_image_numpy(patch, h_del)
image_cible = creation_image_pg(image_cible)
main(image_melange, h_del, image_cible)