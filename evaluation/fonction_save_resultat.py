import csv

# Enregistrement des resultats dans un fichier texte
def Enregistrement(nom_fichier, titre, resultat):
    with open(nom_fichier, mode='w', newline='') as file:
        fichier = csv.writer(file, delimiter=';')
        
        # Ecrire le titre en premiere ligne
        fichier.writerow([titre])
        
        # Ecrire les resultats
        fichier.writerows(resultat)
