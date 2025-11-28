#!/bin/bash

# Spécifiez le GPU à utiliser
GPU=1

# Entraîner les folds 1 et 3
for fold in 1 3
do
    echo "Entraînement du fold $fold sur GPU $GPU"
    #CUDA_VISIBLE_DEVICES=$GPU nnUNet_compile=T nnUNet_n_proc_DA=8 nnUNetv2_train 062 3d_fullres $fold --npz -num_gpus $GPU -tr nnUNetTrainerCELoss_500epochs
    #CUDA_VISIBLE_DEVICES=$GPU nnUNet_compile=T nnUNet_n_proc_DA=8 nnUNetv2_train -tr nnUNetTrainer_1000epochs -p nnUNetPlans_urethra --npz -num_gpus $GPU 062 3d_fullres $fold
    #CUDA_VISIBLE_DEVICES=$GPU nnUNet_compile=T nnUNet_n_proc_DA=8 nnUNetv2_train -tr nnUNetTrainer_1001epochs -p nnUNetPlans --npz -num_gpus $GPU 062 3d_fullres_my_plan $fold
    #CUDA_VISIBLE_DEVICES=$GPU nnUNet_compile=T nnUNet_n_proc_DA=8 nnUNetv2_train -tr nnUNetTrainer_100epochs -p nnUNetPlans --npz -num_gpus $GPU 064 3d_fullres_my_plan $fold
    CUDA_VISIBLE_DEVICES=$GPU nnUNet_compile=T nnUNet_n_proc_DA=8 nnUNetv2_train -tr nnUNetTrainer_50epochs -p nnUNetPlans --npz -num_gpus $GPU 072 3d_fullres_my_plan $fold -pretrained_weights /scratch/nnUNet_results/Dataset072_Prostate/nnUNetTrainer_50epochs__nnUNetPlans__3d_fullres_my_plan/f${fold}/checkpoint_best.pth

    if [ $? -ne 0 ]; then
        echo "Erreur lors de l'entraînement du fold $fold"
        exit 1
    fi
done
