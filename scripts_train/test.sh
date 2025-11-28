#!/bin/bash


# Spécifiez le GPU à utiliser
GPU=1

# Boucle sur les valeurs de fold
for fold in 4
do
    # Chemin du fichier avec fold_0
    original_path="/scratch/nnUNet_results/Dataset062_Prostate/nnUNetTrainer_1001epochs__nnUNetPlans__3d_fullres_my_plan/fold_0/checkpoint_best.pth"

    # Utiliser sed pour remplacer fold_0 par fold_$fold
    modified_path=$(echo $original_path | sed "s/fold_0/fold_$fold/")

    # Afficher le chemin modifié
    echo $modified_path

    CUDA_VISIBLE_DEVICES=$GPU nnUNet_compile=T nnUNet_n_proc_DA=8 nnUNetv2_train -tr nnUNetTrainer_250epochs -p nnUNetPlans --npz -num_gpus $GPU 062 3d_fullres_my_plan $fold -pretrained_weights $modified_path

    if [ $? -ne 0 ]; then
        echo "Erreur lors de l'entraînement du fold $fold"
        exit 1
    fi
done
