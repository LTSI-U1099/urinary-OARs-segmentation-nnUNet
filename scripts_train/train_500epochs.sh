#!/bin/bash

# Spécifiez le GPU à utiliser
GPU=1

# Entraîner les folds 1 et 3
for fold in 0 1 2 3
do
    echo "Entraînement du fold $fold sur GPU $GPU"
    #CUDA_VISIBLE_DEVICES=$GPU nnUNet_compile=T nnUNet_n_proc_DA=8 nnUNetv2_train 062 3d_fullres $fold --npz -num_gpus $GPU -tr nnUNetTrainerCELoss_500epochs
    #CUDA_VISIBLE_DEVICES=$GPU nnUNet_compile=T nnUNet_n_proc_DA=8 nnUNetv2_train -tr nnUNetTrainer_1000epochs -p nnUNetPlans_urethra --npz -num_gpus $GPU 062 3d_fullres $fold
    
    
    original_path="/scratch/nnUNet_results/Dataset062_Prostate/nnUNetTrainer_1001epochs__nnUNetPlans__3d_fullres_my_plan/fold_0/checkpoint_best.pth"

    # Utiliser sed pour remplacer fold_0 par fold_$fold
    modified_path=$(echo $original_path | sed "s/fold_0/fold_$fold/")

    CUDA_VISIBLE_DEVICES=$GPU nnUNet_compile=T nnUNet_n_proc_DA=8 nnUNetv2_train -tr nnUNetTrainer_250epochs -p nnUNetPlans --npz -num_gpus $GPU 062 3d_fullres_my_plan $fold --pretrained_weights $modified_path

    if [ $? -ne 0 ]; then
        echo "Erreur lors de l'entraînement du fold $fold"
        exit 1
    fi
done
