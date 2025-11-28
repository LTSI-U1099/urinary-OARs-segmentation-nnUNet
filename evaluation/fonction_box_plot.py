import csv
import matplotlib.pyplot as plt
import seaborn as sns


######################################################################################################################################
#                                  Cette fonctione fais les boxplot des resultats des metriques                                      #
#                                                                                                                                    #
# Les fichiers contenant les resultats doit etre dans le meme endroit de ce code !                                                   #
#                                                                                                                                    #
# Dans le cas contraire MODIFIER CE CODE pour lui donner le chemin des fichiers avec les resultats                                   #
#                                                                                                                                    #
# Pour ne pas avoir de problemes, le fichier "serveur_nnunet.py" et ce fichier "box_plot.py" les placer dans le meme endroit         #
######################################################################################################################################



# Lecture fichier contenant des donnees
def lire_resultats(nom_fichier):
    resultats = [] # Liste pour stocker les donnees
    with open(nom_fichier, mode="r") as file:   # Ouverture du fichier en mode lecture
        lire = csv.reader(file, delimiter=";")  # Lecture du fichier
        titre = next(lire)[0]                   # Extraire le titre
        structure = next(lire)[1:]              # Extraire les noms des structures (ignorer le premier mot 'Patient')
        for ligne in lire:
            resultats.append([float(x) for x in ligne[1:]])  # Ignorer le numero du patient (qui est l'indice 0)
    return titre, structure, resultats


# Creation box plot
def creer_boxplot(axe, data, labels, ylabel):
    # sns.set_theme(style="darkgrid")
    # axe.violinplot(data,showmedians=True)
    axe.boxplot(data)
    
    # sns.stripplot(data)
    axe.set_xticklabels(labels)
    axe.set_ylabel(ylabel)
    
    # Inclinaison des étiquettes de l'axe des x
    plt.setp(axe.get_xticklabels(), rotation=45, ha='right')


# Fonction pour créer et sauvegarder un boxplot individuel
def enregistrer_boxplot(donnees, nom_structure, titre_graphique, nom_fichier):
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle(titre, fontsize=14)
    creer_boxplot(ax, donnees, nom_structure, titre_graphique)
    fig.tight_layout()
    plt.savefig(nom_fichier, format='jpeg')
    plt.close(fig)



###################################################################
# ---------- Chemins vers les fichiers de donnees ----------
#chemin = "/chemin/a/choisir/exemple"
#titre, nom_structure, resultats_dice = lire_resultats(chemin)
#_, _, resultats_precision = lire_resultats(chemin)
# etc

###################################################################


chemin_base = f"/home/mcastro/val_40_49_50epochs_new_plan_it1_pp/"
chemin_base = f"/home/mcastro/prostatex_50epochs_new_plan_it1v1/"
# Le titre et le noms des structures sont toujours les meme, _, _, les ignore (car pas besoin)
titre, nom_structure, resultats_dice = lire_resultats(chemin_base+"dice_results_3d.txt")
_, _, resultats_precision = lire_resultats(chemin_base+"precision_results_3d.txt")
_, _, resultats_recall = lire_resultats(chemin_base+"recall_results_3d.txt")
_, _, resultats_hausdorff = lire_resultats(chemin_base+"hausdorff_results_3d.txt")
_, _, resultats_volume = lire_resultats(chemin_base+"volume_results_3d.txt")

_, _, resultats_mean_surface = lire_resultats(chemin_base+"mean_surface_3d.txt")
_, _, resultats_relative_volume_difference = lire_resultats(chemin_base+"relative_volume_difference_3d.txt")
_, _, resultats_volumetric_overlap_error = lire_resultats(chemin_base+"volumetric_overlap_error_3d.txt")

# ---------- Conversion des resultats au format compatible pour les box plots ----------
# "Regrouper" les donnees de chaque structure
resultats_dice_boxplot = list(map(list, zip(*resultats_dice)))
resultats_precision_boxplot = list(map(list, zip(*resultats_precision)))
resultats_recall_boxplot = list(map(list, zip(*resultats_recall)))
resultats_hausdorff_boxplot = list(map(list, zip(*resultats_hausdorff)))
resultats_volume_boxplot = list(map(list, zip(*resultats_volume)))
resultats_mean_surface_boxplot = list(map(list, zip(*resultats_mean_surface)))
resultats_relative_volume_difference_boxplot = list(map(list, zip(*resultats_relative_volume_difference)))
resultats_volumetric_overlap_error_boxplot = list(map(list, zip(*resultats_volumetric_overlap_error)))

# --------------- Creation des box plots ---------------
# Création et sauvegarde de chaque graphique individuellement
titre = "Score DICE distribution for each organ"
enregistrer_boxplot(resultats_dice_boxplot, nom_structure, "Score DICE", chemin_base+"dice_score.jpeg")
titre = "Hausdorff Distance distribution for each organ"
enregistrer_boxplot(resultats_hausdorff_boxplot, nom_structure, "Hausdorff Distance", chemin_base+"hausdorff_distance.jpeg")
titre = "Score Precision distribution for each organ"
enregistrer_boxplot(resultats_precision_boxplot, nom_structure, "Precision", chemin_base+"precision.jpeg")
titre = "Score RECALL distribution for each organ"
enregistrer_boxplot(resultats_recall_boxplot, nom_structure, "Recall", chemin_base+"recall.jpeg")
titre = "Volume distribution for each organ"
enregistrer_boxplot(resultats_volume_boxplot, nom_structure, "Volume", chemin_base+"volume.jpeg")

titre = "Mean Surface Distance for each organ"
enregistrer_boxplot(resultats_mean_surface_boxplot, nom_structure, "Mean Surface Distanc", chemin_base+"mean_surface.jpeg")
titre = "Relative Volume Difference for each organ"
enregistrer_boxplot(resultats_relative_volume_difference_boxplot, nom_structure, "Relative Volume Difference", chemin_base+"relative_volume_difference.jpeg")
titre = "Volumetric Overlap Error for each organ"
enregistrer_boxplot(resultats_volumetric_overlap_error_boxplot, nom_structure, "Volumetric Overlap Error", chemin_base+"volumetric_overlap_error.jpeg")
#### Figure 1 : Dice Score et Hausdorff
#
#fig1, axe1 = plt.subplots(2, 1, figsize=(14, 6))
## Titre general
#fig1.suptitle(titre, fontsize=16)
#
## Dice Score
#creer_boxplot(axe1[0], resultats_dice_boxplot, nom_structure, "Dice Score")
## Distance de Hausdorff
#creer_boxplot(axe1[1], resultats_hausdorff_boxplot, nom_structure, "Hausdorff Distance")
#
#fig1.tight_layout()# Ajuster auto
##plt.show()
#plt.savefig('graphique1.jpeg', format='jpeg')
## --------------------------------------------------
##### Figure 2 : Precision et Rappel
#
#fig2, axe2 = plt.subplots(3, 1, figsize=(14, 6))
#
## Titre general
#fig2.suptitle(titre, fontsize=16)
#
## Precision
#creer_boxplot(axe2[0], resultats_precision_boxplot, nom_structure, "Precision")
## Rappel
#creer_boxplot(axe2[1], resultats_recall_boxplot, nom_structure, "Rappel")
## volume
#creer_boxplot(axe2[2], resultats_volume_boxplot, nom_structure, "Volume")
#fig2.tight_layout() # Ajuster auto
#
##plt.show()
#plt.savefig('graphique2.jpeg', format='jpeg')
