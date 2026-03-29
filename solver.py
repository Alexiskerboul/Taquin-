import heapq

class SolverTaquin():
    def __init__(self, h_del, etat_initial, etat_cible=(0, 1, 2, 3, 4, 5, 6, 7, 8)):
        self.etat_initial = etat_initial
        self.etat_cible = etat_cible
        self.h_del = h_del
        self.taille = 3

    def coups_possibles(self, etat):
        """
        Renvoie l'ensemble des états possibles après un coup à partir d'un état initial
        etat est un TUPLE 
        """
        index_vide = etat.index(self.h_del)
        i_del = index_vide // 3
        j_del = index_vide % 3
        direction = [(0,1), (0,-1), (1, 0), (-1,0)]
        voisins = []
        for i, j in direction:
            i_new = i_del + i
            j_new = j_del +j
            if 0 <= i_new < 3 and 0 <= j_new < 3:
                h_new = 3*i_new + j_new
                new_etat = list(etat)
                new_etat[h_new], new_etat[index_vide] = new_etat[index_vide], new_etat[h_new]
                voisins.append(tuple(new_etat))
        return voisins
    
    def heuristique(self, etat):
        """
        Renvoie notre heuristique (distance de Manhattan)
        """
        distance_totale = 0
        for tuile in range(self.taille**2):
            if tuile != self.h_del:
                index_current = etat.index(tuile)
                i_current = index_current //3
                j_current = index_current % 3
                index_cible = self.etat_cible.index(tuile)
                i_cible = index_cible // 3
                j_cible = index_cible % 3
                distance = abs(i_current - i_cible) + abs(j_cible - j_current)
                distance_totale += distance
        return distance_totale
    
    def solve(self):
        """
        Renvoie le chemin le plus court allant de l'état initial à l'état final en appliquant A*
        """
        open_list = []
        f_debut = self.heuristique(self.etat_initial)
        heapq.heappush(open_list, (f_debut, self.etat_initial, [self.etat_initial]))
        g_scores = {self.etat_initial:0}
        closed_list = set()
        while open_list:
            f_courant, etat_courant, chemin = heapq.heappop(open_list)
            if etat_courant == self.etat_cible:
                print(f"Solution trouvée en {len(chemin) - 1} coups !")
                return chemin
            closed_list.add(etat_courant)
            for voisin in self.coups_possibles(etat_courant):
                if voisin in closed_list:
                    continue
                new_g = g_scores[etat_courant] + 1 
                if voisin not in g_scores or new_g < g_scores[voisin]:
                    g_scores[voisin] = new_g
                    f_voisin = new_g + self.heuristique(voisin)
                    nouveau_chemin = chemin + [voisin] 
                    heapq.heappush(open_list, (f_voisin, voisin, nouveau_chemin))
        print("Aucune solution trouvée. Ce taquin est mathématiquement insoluble.")
        return None
