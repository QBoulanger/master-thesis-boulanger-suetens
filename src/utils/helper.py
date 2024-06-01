import random


def generer_liste_de_tuples(taille, x, y):
    liste_de_tuples = []
    for _ in range(taille):
        a = random.randint(x, y)
        b = random.randint(x, y)
        liste_de_tuples.append((a, b))
    return liste_de_tuples


# Exemple d'utilisation
taille_A = 12
borne_inf = 1
borne_sup = 12
resultat = generer_liste_de_tuples(taille_A, borne_inf, borne_sup)
print("Liste générée de tuples :", resultat)
