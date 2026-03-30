import pygame as pg 
from dataset import DatasetPuzzle
from torch.utils.data import DataLoader
from utils import creation_image_numpy, creation_image_pg, change_place
import random
import numpy as np 
from solver import SolverTaquin

def main(image, h_del, image_cible, permutation, patch):
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
    en_resolution = False
    chemin_automatique = []
    index_etape = 0
    temps_dernier_coup = 0
    delai_animation = 700
    etat_logique = tuple(int(x) for x in permutation)
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
                if event.key == pg.K_SPACE and not en_resolution:
                    compteur = 0
                    solver = SolverTaquin(h_del=h_del, etat_initial=etat_logique, etat_cible=tuple(range(9)))
                    chemin = solver.solve()
                    if chemin:
                        chemin_automatique = chemin
                        en_resolution = True
                        index_etape = 1  
                        temps_dernier_coup = pg.time.get_ticks()
        if en_resolution:
            temps_actuel = pg.time.get_ticks()
            if temps_actuel - temps_dernier_coup > delai_animation:
                if index_etape < len(chemin_automatique):
                    etat_actuel = chemin_automatique[index_etape]
                    elem_entiers = [t.item() if hasattr(t, 'item') else t for t in etat_actuel]
                    position_case_noire = elem_entiers.index(h_del)
                    patch_melange = patch[elem_entiers]
                    image_melange = creation_image_numpy(patch_melange, position_case_noire)
                    image_pg = creation_image_pg(image_melange)
                    surface_temporaire = pg.surfarray.make_surface(image_pg)
                    surface_agrandie = pg.transform.scale(surface_temporaire, taille_fenetre)
                    index_etape += 1
                    compteur += 1
                    temps_dernier_coup = temps_actuel  
                else:
                    en_resolution = False
                    partie_gagnee = True 
        ecran.blit(surface_agrandie, (0, 0))
        texte_compteur = police_compteur.render(f" Coups : {compteur} ", True, (255, 255, 255), (0, 0, 0))
        ecran.blit(texte_compteur, (10, 10))
        if partie_gagnee:
            fond_sombre = pg.Surface(taille_fenetre)
            fond_sombre.set_alpha(150) 
            fond_sombre.fill((0, 0, 0))
            ecran.blit(fond_sombre, (0, 0))
            police = pg.font.SysFont(None, 32)
            texte_victoire = police.render(f"Félicitations ! Gagné en {compteur} coups !", True, (50, 200, 50))
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
main(image_melange, h_del, image_cible, permutation, patch)