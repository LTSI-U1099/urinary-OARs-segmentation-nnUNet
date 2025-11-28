# ---------- Bibliotheques ----------
import nibabel as nib       # lecture et ecriture de donnees 
import numpy as np
import SimpleITK as sitk    # traitement images medicales avancees
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import csv
import os


# ---------- Mes Fonctions ----------
# from fichier principal import la fonction
from fonction_metric_dice_score import dice_score 
from fonction_metric_precision import precision
from fonction_metric_rappel import recall
from fonction_metric_hausdorff import hausdorff_distance
from fonction_metric_volume import calculate_volumes
from fonction_metric_volume import mean_surface_distance
from fonction_metric_volume import relative_volume_difference
from fonction_metric_volume import volumetric_overlap_error
from fonction_save_resultat import Enregistrement


#################################### A modifier  #################################
# --------------------- Choisir les noms des organes

# ---------- Fonction pour creer une liste contenant les noms les differentes structures ----------
        
        ##############################################################
        #     Ajouter des nouvelles listes pour d'autres modele      #
        ##############################################################

def liste_structure(a):
    
    if a == "prostate":
        return [["Pacient","Background", "Corps", "Os", "Prostate", "Vessie", "Rectum"]]
    
    elif a == "petite_structure" :
        return [["Pacient","Background", "Bladderneck", "Bulbousurethra", "Intraprostaticurethra", "Membranousurethra", "Ureters", "Striatedsphincter"]]
    
    elif a == "structure_Stria_membra" :
        return [["Pacient","Background", "Bladdertrigone", "Bulbousurethra", "Intraprostaticurethra", "Stria_membra", "Ureters"]]
    
    elif a == "petite_structure_new" :
        return [["Pacient","Background","Bladder","Prostate","Intraprostaticurethra", "Rectum", "Ureters", "Bladderneck", "Bladdertrigone", "Bulbousurethra", "Mmbranousurethra", "Internecktrigone"]]

        

##################################################################################
###### Choisir la structure dans liste_structure() !!!!!!!!!!

# Liste pour enregistrer les resultats
structure = liste_structure("petite_structure_new")[0]  # Choix dela strcuture (doit correspondent a la fonction liste_structure())
structure.remove("Background") # PAS DE MODIFICATION ICI !!!
structure.remove("Internecktrigone") 
# Liste pour la legende des histogrammes
labels = liste_structure("petite_structure_new")[0]     # Choix dela strcuture (doit correspondent a la fonction liste_structure()) 
labels.remove("Pacient")   # PAS DE MODIFICATION ICI !!!

##################################################################################


# ---------- Listes pour stocker les resultats des scores ----------
# Pours les images 3D
results_dice_image = [structure]
results_precision_image = [structure]
results_recall_image = [structure]
results_hausdorff_image = [structure]
results_volume_image = [structure]
results_vol_ref_image = [structure]
results_mean_surface_distance_image = [structure]
results_relative_volume_difference_image = [structure]
results_volumetric_overlap_error_image = [structure]

# Pour les 3 coupes (image 2D)
results_dice_trans = [structure]
results_dice_sagittal = [structure]
results_dice_coronal = [structure]

results_precision_trans = [structure]
results_precision_sagittal = [structure]
results_precision_coronal = [structure]

results_recall_trans = [structure]
results_recall_sagittal = [structure]
results_recall_coronal = [structure]

results_hausdorff_trans = [structure]
results_hausdorff_sagittal = [structure]
results_hausdorff_coronal = [structure]

# -------------------- Boucle pour charger toutes les images --------------------
# for i in [1, 2, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,23,24,25,26,27,28,29,30,34, 35, 36, 37, 38, 39, 40]: # range(1, 54):  #  (si le nombre de patients/donnees correspond 53  ==> range(1,54))
#for i in [5, 6, 7, 8, 9, 10, 11]: # range(1, 54):  #  (si le nombre de patients/donnees correspond 53  ==> range(1,54))
#for i in [15, 16, 17, 1, 4, 5, 6]:
# for i in [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]:
for i in [0,19,22,27,38,49,51,56,76,81,92,109,119,128,141,144,152,153,175,176,186,195,198,201,250,266,269,272,282,297,317,319,323,327,336,415,419,544,575,576,586,598,645,652,669,670,672,727]:
    x = "{:03d}".format(i)  # Format trois chiffres et rempli de zero pour completer le format a 3 chiffre !!! NE PAS MODIFIER !!!

    # ------ Chemins fichiers .nii ------ !!!!!! A MODIFIER !!!!!!
    #chemin_imagesTr = ("/home/mcastro/Documents/nnUNet/nnUNet_raw/Dataset062_Prostate/imagesTr/CEM_"+x+"_0000.nii.gz")
    #chemin_labelsTr = ("/home/mcastro/Documents/nnUNet/nnUNet_raw/Dataset062_Prostate/labelsTr/CEM_"+x+".nii.gz")
    chemin_labelsTr = ("/home/mcastro/validation_in/CEM_"+x+"_out.nii.gz")
    chemin_labelsTr = ("/home/mcastro/val_prostatex/pro_"+x+".nii.gz")

    fold_number = 9 #'all' # Remplacez cette valeur par la variable souhaitée
    #chemin_base = f"/home/mcastro/Documents/nnUNet/valCEM40_49_finetuning{fold_number}/"
    #chemin_base = f"/home/mcastro/Documents/nnUNet/validation_pretrain/"
    chemin_base = f"/home/mcastro/val_40_49_50epochs_new_plan_it1_pp/"
    chemin_base = f"/home/mcastro/prostatex_50epochs_new_plan_it1v1/"

    # chemin_modele = f"/home/mcastro/Documents/nnUNet/valCEM40_49_finetuning{fold_number}/CEM_{x}.nii.gz"
    chemin_modele = f"/home/mcastro/val_40_49_50epochs_new_plan_it1_pp/CEM_{x}_out.nii.gz"
    chemin_modele = f"/home/mcastro/prostatex_50epochs_new_plan_it1v1/pro_{x}.nii.gz"
    #chemin_modele = f"{chemin_base}CEM_{x}.nii.gz"
    # titre_txt = f"Test finetuning Fold {fold_number} on CEMs patients"
    titre_txt = f"Cross validation 50epochs new_plan train on CEM and Dijon patients"
#######################################################################################################

    # Charger les images .nii
    # imagesTr = sitk.ReadImage(chemin_imagesTr)  # images IRM
    labelsTr = sitk.ReadImage(chemin_labelsTr)  # images segmenter (par medecin, verite terrain)
    modele = sitk.ReadImage(chemin_modele)      # image segmenter par UNet
    
    # Obtenir la taille des voxels
    voxel_size = modele.GetSpacing()

    # Obtenir les donner des images
    # donnees_IRM = sitk.GetArrayFromImage(imagesTr)          # IRM                                   
    donnees_segmenter = sitk.GetArrayFromImage(labelsTr)    # Segmentation medecin / Verite terrain 
    donnees_modele = sitk.GetArrayFromImage(modele)         # Segmentation Modele (UNet)            

     # -------------------- Calcul des metriques des images (3D) --------------------
    max_val = len(labels)-1 # Combien de structures
    dice_score_image = dice_score(donnees_segmenter, donnees_modele, max_val)
    prec_image = precision(donnees_segmenter, donnees_modele, max_val)
    rappel_image = recall(donnees_segmenter, donnees_modele, max_val)
    hausdorff_image = hausdorff_distance(donnees_segmenter, donnees_modele, max_val)
    volume_score, vol_ref = calculate_volumes(donnees_segmenter, donnees_modele, max_val, voxel_size)
    msd = mean_surface_distance(donnees_segmenter, donnees_modele, max_val, voxel_size)
    rvd = relative_volume_difference(donnees_segmenter, donnees_modele, max_val)
    voe = volumetric_overlap_error(donnees_segmenter, donnees_modele, max_val)

    results_dice_image.append([i, *dice_score_image])
    results_precision_image.append([i, *prec_image])
    results_recall_image.append([i, *rappel_image])
    results_hausdorff_image.append([i, *hausdorff_image])
    results_volume_image.append([i, *volume_score])
    results_vol_ref_image.append([i, *vol_ref])
    results_volumetric_overlap_error_image.append([i, *voe])   
    results_relative_volume_difference_image.append([i, *rvd])
    results_mean_surface_distance_image.append([i, *msd])

    print("Patient %d fini" % i)

# ---------- Enregistrement des resultats dans des fichiers texte ----------
# Images 3D
Enregistrement(f"{chemin_base}dice_results_3d.txt", titre_txt, results_dice_image)
print("Enregistrement Dice fini")

Enregistrement(f"{chemin_base}precision_results_3d.txt", titre_txt, results_precision_image)
print("Enregistrement Precision fini")

Enregistrement(f"{chemin_base}recall_results_3d.txt", titre_txt, results_recall_image)
print("Enregistrement Recall fini")

Enregistrement(f"{chemin_base}hausdorff_results_3d.txt", titre_txt, results_hausdorff_image)
print("Enregistrement Hausdorff fini")

Enregistrement(f"{chemin_base}volume_results_3d.txt", titre_txt, results_volume_image)
Enregistrement(f"{chemin_base}volume_refs_3d.txt", titre_txt, results_vol_ref_image)

Enregistrement(f"{chemin_base}mean_surface_3d.txt", titre_txt, results_mean_surface_distance_image)
Enregistrement(f"{chemin_base}relative_volume_difference_3d.txt", titre_txt, results_relative_volume_difference_image)
Enregistrement(f"{chemin_base}volumetric_overlap_error_3d.txt", titre_txt, results_volumetric_overlap_error_image)

print("Enregistrement volume fini")
