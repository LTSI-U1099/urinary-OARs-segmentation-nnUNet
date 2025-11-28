# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:29:08 2025

@author: Miguel Castro USER
"""

import os
import shutil
import nibabel as nib
import numpy as np

def merge_masks(directory):
    # Définir la correspondance des masques avec leurs valeurs respectives
    mask_mapping = {
     "mask_Rectum_ext": 4,
     "mask_bladder": 1,
     "mask_Prostate": 2,
     "mask_intraprostaticurethra": 3,
     "mask_ureters": 5,
     "mask_bladderneck": 7,
     "mask_bladdertrigone": 6,
     "mask_bulbousurethra": 8,
     "mask_membranousurethra": 9
    }

    # Initialiser un dictionnaire vide pour stocker les données des masques
    mask_data = {}

    # Charger chaque fichier de masque et vérifier les intersections
    for mask_name, mask_value in mask_mapping.items():
        file_name = os.path.join(directory, f"{mask_name}.nii.gz")
        
        if os.path.exists(file_name):
            img = nib.load(file_name)
            data = img.get_fdata()
            
            if np.any(data > 0):
                if mask_data:
                    intersection = np.any(np.isin(data, list(mask_data.values())))
                    if intersection:
                        print(f"Intersection found in {file_name}")
                mask_data[mask_name] = data * mask_value
        else:
            print(f"file not found in {file_name}")
    # Fusionner tous les masques en un seul tableau en conservant les étiquettes d'origine
    merged_data = np.zeros_like(list(mask_data.values())[0])
    for mask_name, data in mask_data.items():
        merged_data[data > 0] = mask_mapping[mask_name]

    # Créer une nouvelle structure avec l'étiquette 10 pour l'intersection de "mask_Bladderneck" et "mask_Bladdertrigone"
    if "mask_bladderneck" in mask_data and "mask_bladdertrigone" in mask_data:
        bladderneck_data = mask_data["mask_bladderneck"]
        bladdertrigone_data = mask_data["mask_bladdertrigone"]
        
        # Vérifier l'intersection
        intersection_data = (bladderneck_data > 0) & (bladdertrigone_data > 0)
        if np.any(intersection_data):
            print("Intersection found between mask_Bladderneck and mask_Bladdertrigone")
            merged_data[intersection_data] = 10

    return merged_data, img.affine

def process_patients(input_directory, output_directory):
    # Créer les dossiers de sortie s'ils n'existent pas
    images_output_dir = os.path.join(output_directory, 'imagesTr')
    labels_output_dir = os.path.join(output_directory, 'labelsTr')
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(labels_output_dir, exist_ok=True)

    # Parcourir les sous-dossiers des patients
    for subdir in os.listdir(input_directory):
        if subdir.startswith("Patient "):
            patient_number = subdir.split(" ")[1]
            patient_dir = os.path.join(input_directory, subdir)
            print("Patient ", patient_number)

            # Copier le fichier image.nii.gz dans le dossier de sortie images
            #image_file_src = os.path.join(patient_dir, 'image.nii.gz')
            #image_file_dst = os.path.join(images_output_dir, f'CEM_0{patient_number}_0000.nii.gz')
            #shutil.copy(image_file_src, image_file_dst)

            # Fusionner les masques et enregistrer le résultat dans le dossier de sortie labels
            merged_data, affine = merge_masks(patient_dir)
            merged_img = nib.Nifti1Image(merged_data, affine)
            label_file_dst = os.path.join(labels_output_dir, f'CEM_0{patient_number}.nii.gz')
            nib.save(merged_img, label_file_dst)

# Exemple d'utilisation
input_directory = 'D:\\CEM\\new'  # Remplacez par le chemin de votre dossier d'entrée contenant les sous-dossiers des patients
output_directory = 'D:\CEM_nnUNet'  # Remplacez par le chemin de votre dossier de sortie
process_patients(input_directory, output_directory)


