# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 09:43:44 2025

@author: Miguel Castro USER
"""

# import os
# import SimpleITK as sitk
# import numpy as np

# def resample_image(image, out_spacing, out_size, out_origin, out_direction, is_label=False):
#     resample = sitk.ResampleImageFilter()
#     resample.SetOutputSpacing(out_spacing)
#     resample.SetSize(out_size)
#     resample.SetOutputDirection(out_direction)
#     resample.SetOutputOrigin(out_origin)
#     resample.SetTransform(sitk.Transform())
#     resample.SetDefaultPixelValue(image.GetPixelIDValue())
#     resample.SetInterpolator(sitk.sitkLinear if not is_label else sitk.sitkNearestNeighbor)
#     return resample.Execute(image)

# # --- Paramètres ---
# images_dir = 'G:\prostate BDs\PROSTATEx-files\Dataset064_ensemble_harmonise/imagesTr'  # images à corriger (ex : PROSTATEx, Dijon)
# output_dir = 'G:\prostate BDs\PROSTATEx-files\Dataset065_ensemble_harmonise/imagesTr'
# os.makedirs(output_dir, exist_ok=True)

# reference_dir = "G:\prostate BDs\PROSTATEx-files\images"
# reference_files = [os.path.join(reference_dir, f)
#                    for f in os.listdir(reference_dir)
#                    if (f.endswith('.nii') or f.endswith('.nii.gz'))]

# print(f"Fichiers de référence trouvés : {len(reference_files)}")
# if not reference_files:
#     raise ValueError("Aucun fichier de référence CEM trouvé dans le dossier spécifié.")

# # --- 1. Déterminer le spacing cible (ici, celui de la première image CEM) ---
# try:
#     first_img = sitk.ReadImage(reference_files[0])
# except Exception as e:
#     raise RuntimeError(f"Impossible de lire la première image de référence : {reference_files[0]}. Erreur: {e}")

# target_spacing = first_img.GetSpacing()
# target_size = first_img.GetSize()
# target_origin = first_img.GetOrigin()
# target_direction = first_img.GetDirection()

# print(f"Target spacing: {target_spacing}, Target size: {target_size}")

# # --- 2. Charger et rééchantillonner toutes les images CEM ---
# arrays = []
# for f_path in reference_files:
#     try:
#         img = sitk.ReadImage(f_path)
#         # Assure-toi que les dimensions de l'image source sont 3D
#         if len(img.GetSize()) != 3:
#             print(f"Avertissement: L'image {os.path.basename(f_path)} n'est pas 3D ({img.GetSize()}). Elle sera ignorée.")
#             continue
        
#         img_resampled = resample_image(img, target_spacing, target_size, target_origin, target_direction)
#         arrays.append(sitk.GetArrayFromImage(img_resampled).astype(np.float32))
#         print(f"Image {os.path.basename(f_path)} rééchantillonnée. Forme du tableau : {sitk.GetArrayFromImage(img_resampled).shape}")
#     except Exception as e:
#         print(f"Erreur lors du traitement de l'image {os.path.basename(f_path)}: {e}. Cette image sera ignorée.")

# print(f"Nombre total d'images CEM traitées avec succès : {len(arrays)}")

# if not arrays:
#     raise ValueError("Aucune image CEM n'a pu être traitée avec succès pour calculer la moyenne. Vérifiez vos fichiers.")

# mean_array = np.mean(arrays, axis=0)
# print(f"Dimensions du tableau moyen : {mean_array.shape}")

# # --- 3. Reconstruire l'image moyenne SimpleITK ---
# try:
#     reference_image = sitk.GetImageFromArray(mean_array)
#     reference_image.SetSpacing(target_spacing)
#     reference_image.SetOrigin(target_origin)
#     reference_image.SetDirection(target_direction)
# except Exception as e:
#     raise RuntimeError(f"Erreur lors de la création de l'image de référence à partir du tableau moyen : {e}")

# sitk.WriteImage(reference_image, os.path.join(output_dir, "reference_moyenne_CEM.nii.gz")) # Sauvegarder dans output_dir pour être sûr

# # --- 4. Harmoniser les autres images par histogram matching ---
# # (Reste inchangé par rapport à ton code précédent)
# for fname in os.listdir(images_dir):
#     if not (fname.endswith('.nii') or fname.endswith('.nii.gz')):
#         continue
#     img_path = os.path.join(images_dir, fname)
#     img = sitk.ReadImage(img_path)
    
#     # Vérifie que l'image à harmoniser est 3D
#     if len(img.GetSize()) != 3:
#         print(f"Avertissement: L'image à harmoniser {os.path.basename(img_path)} n'est pas 3D ({img.GetSize()}). Elle sera ignorée.")
#         continue

#     matched = sitk.HistogramMatching(
#         img, reference_image,
#         numberOfHistogramLevels=100,
#         numberOfMatchPoints=50,
#         thresholdAtMeanIntensity=True
#     )
#     sitk.WriteImage(matched, os.os.path.join(output_dir, fname))

# print("Histogram matching terminé. Les images sont harmonisées sur la distribution d'intensité moyenne CEM (1.5T).")

import os
import SimpleITK as sitk

def resample_image(image, out_spacing, out_size, out_origin, out_direction, is_label=False):
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(out_direction)
    resample.SetOutputOrigin(out_origin)
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkLinear if not is_label else sitk.sitkNearestNeighbor)
    return resample.Execute(image)

# --- Paramètres ---
# images_dir = 'G:/prostate BDs/PROSTATEx-files/Dataset064_ensemble_harmonise/imagesTr'
# output_dir = 'G:/prostate BDs/PROSTATEx-files/Dataset065_ensemble_harmonise/imagesTr'
images_dir = 'G:/Seg_prostate/corrections/imageP'
output_dir = 'G:/Seg_prostate/corrections/labelsP'
os.makedirs(output_dir, exist_ok=True)

# --- Charger l'image de référence déjà créée ---
reference_image_path = 'G:/prostate BDs/PROSTATEx-files/Dataset065_ensemble_harmonise/imagesTr/reference_moyenne_CEM.nii.gz'
reference_image = sitk.ReadImage(reference_image_path)
reference_image = sitk.Cast(reference_image, sitk.sitkFloat32)  # Pour compatibilité mémoire et HistogramMatching

target_spacing = reference_image.GetSpacing()
target_size = reference_image.GetSize()
target_origin = reference_image.GetOrigin()
target_direction = reference_image.GetDirection()

print(f"Référence: size={target_size}, spacing={target_spacing}")

# --- Harmoniser toutes les images ---
for fname in os.listdir(images_dir):
    if not (fname.endswith('.nii') or fname.endswith('.nii.gz')):
        continue
    img_path = os.path.join(images_dir, fname)
    img = sitk.ReadImage(img_path)
    if len(img.GetSize()) != 3:
        print(f"Avertissement: {fname} n'est pas 3D ({img.GetSize()}). Ignorée.")
        continue

    img = sitk.Cast(img, sitk.sitkFloat32)
    # Rééchantillonner à la taille et au spacing de la référence
    # img_resampled = resample_image(
    #     img,
    #     target_spacing,
    #     target_size,
    #     target_origin,
    #     target_direction
    # )

    matched = sitk.HistogramMatching(
        img, reference_image,    
        numberOfHistogramLevels=100,
        numberOfMatchPoints=50,
        thresholdAtMeanIntensity=True
    )
    sitk.WriteImage(matched, os.path.join(output_dir, fname))

    
    # matched = sitk.HistogramMatching(
    #     img_resampled, reference_image,
    #     numberOfHistogramLevels=100,
    #     numberOfMatchPoints=50,
    #     thresholdAtMeanIntensity=True
    # )
    # sitk.WriteImage(matched, os.path.join(output_dir, fname))
    print(f"{fname} harmonisée et sauvegardée.")

print("Toutes les images ont été rééchantillonnées à (370, 768, 768) et harmonisées (histogram matching) en float32.")
