# urinary-OARs-segmentation-nnUNet

## Overview
This repository hosts the code, trained models, and scripts for a deep learning method based on nnU-Net, designed for automatic segmentation of urinary organs at risk (OARs) and pelvic structures on multicentric prostate MRI. The method targets key structures such as the intraprostatic urethra, bladder trigone, bladder neck, bulbous urethra, ureters, bladder, rectum, and prostate for prostate cancer radiotherapy planning.

The model is trained and validated on datasets from multiple centers including MR-linac devices (Unity®, Elekta and MRIdian®, Viewray) and the public [PROSTATEx](https://www.cancerimagingarchive.net/collection/prostatex/) database (Siemens MAGNETOM Trio and Skyra MR scanners). Evaluation metrics include Dice Similarity Coefficient (DSC), Hausdorff Distance (HD95), and Surface Distance (SD).

This repository accompanies an ongoing, under-review study on automatic segmentation of urinary organs at risk for prostate cancer radiotherapy and does not yet correspond to a final, published article.

## Built on nnU-Net
This project builds upon the powerful and self-configuring [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) framework for medical image segmentation. We implemented a customized 3D full-resolution nnU-Net configuration optimized for segmenting small urinary structures while maintaining anatomical context for larger pelvic organs.

## Data and Labels

- Multicentric MRI datasets:
  - MR-linac images from Unity® and MRIdian®
  - [PROSTATEx](https://www.cancerimagingarchive.net/collection/prostatex/)x public dataset from Siemens
- Imaging modality: T2-weighted MRI

The labels are derived from manual contours. In some cases, there are overlaps between contours, particularly for labels 8 (bulbous urethra) and 9 (membranous urethra). Therefore, a preprocessing step is required to convert the contours from RTstruct format to NIfTI files before using the segmented images in the CNN. After the segmentation results are obtained, a postprocessing step is necessary to reconstruct the bulbous urethra and membranous urethra structures.

- Label set:
  - 0: background
  - 1: bladder
  - 2: prostate
  - 3: intraprostaticurethra
  - 4: rectum
  - 5: ureters
  - 6: bladderneck
  - 7: bladdertrigone
  - 8: 8: bulbousurethra (This region is the part of bulbous urethra that does not include membranous urethra)
  - 9: membranousurethra (This region is the part of membranous urethra that does not include bulbous urethra)
  - 10: internecktrigone (This region is where bulbous urethra and membranous urethra overlap)
  
![Diagramme](documents/figs/Screenshot1.jpg)

## Repository Structure

```
/model/
/documents/
/trainers/               \# Custom nnU-Net trainer scripts
/nnUNetPlans/            \# Custom nnU-Net plans JSON files
/scripts/
/prepare_data/           \# Scripts for data conversion and dataset organization
/training/               \# Custom nnU-Net trainer scripts
/inference/              \# Scripts for inference and RTStruct conversion
```

## Installation and Requirements

- Python 3.9+
- PyTorch with CUDA support
- nnU-Net version 2 (Please follow the official nnU-Net repository for detailed and up-to-date installation instructions: https://github.com/MIC-DKFZ/nnUNet)
- SimpleITK, nibabel, medpy, pydicom, rt-utils
  
## Data Preparation

- Follow nnU-Net dataset folder conventions:

```

Dataset072_Prostate/
imagesTr/
labelsTr/
imagesTs/ (optional)
dataset.json

```

- RTstruct contours to NIfTI masks:

Convert RTstruc contours to NIfTI masks using [dcm2niix](https://github.com/rordenlab/dcm2niix?tab=readme-ov-file).
To prepare the dataset, the RTstruc DICOM must be converted to NIfTI format. This can be done using the [dcm2niix](https://github.com/rordenlab/dcm2niix?tab=readme-ov-file) tool. Below is an example command:
```

dcm2niix.exe -o "C:\output_folder" -f "%p_%s" -z y "C:\input_folder"

```
We used a Python script that leverages the [dcm2niix Python library](https://pypi.org/project/dcm2niix/) to convert RTstruct files into NIfTI format. Below is an example script:

```

python prepare_data/script_dcmrtstruct2nii.py 

```

The previous routine generates a folder containing individual segmentation masks in NIfTI format. The script merge_masks.py consolidates these masks into a single NIfTI segmentation file, assigning the appropriate labels as described earlier.
To ensure accurate representation of overlapping regions, the original bulbous urethra and membranous urethra masks are split into three distinct masks:

- bulbousurethra: the portion of the bulbous urethra excluding the membranous urethra.
- membranousurethra: the portion of the membranous urethra excluding the bulbous urethra.
- internecktrigone: a new mask representing the overlapping region between the bulbous urethra and membranous urethra.

```

python prepare_data/merge_masks.py 

```

- Harmonize multi-center data via resampling, N4 bias correction, anisotropic filtering, histogram matching, etc. (see scripts prepare_data/harmonize_data.py)

```

python prepare_data/harmonize_data.py 

```

## Training

- Plan and preprocess data:

```

nnUNetv2_plan_and_preprocess -d 072 -c 3d_fullres --verify_dataset_integrity

```


- Train 5-fold cross-validation models (example for fold 0):

```
CUDA_VISIBLE_DEVICES=0 nnUNet_compile=T nnUNet_n_proc_DA=8 nnUNetv2_train -tr nnUNetTrainer_1000epochs -p nnUNetPlans --npz -num_gpus 0 072 3d_fullres_my_plan 0
```

- Repeat for folds 1 to 4 on available GPUs.

- Training parameters:
  - Patch size and batch size optimized for small structure segmentation
  - Loss: combination of Dice and Cross-Entropy (optionally boundary loss)
  - Extensive data augmentation applied

We have prepared scripts that automate training on two GPUs and enable fine-tuning from existing checkpoints instead of starting from scratch. These scripts are available in the /prepare_data folder.

## Inference

Automatically determine the best configuration
Once the desired configurations were trained (full cross-validation) you can tell nnU-Net to automatically identify the best combination for you:

```

nnUNetv2_find_best_configuration 072 -c 3d_fullres -tr nnUNetTrainer_1000epochs 

```

- Run inference on test data:

```

CUDA_VISIBLE_DEVICES=0  nnUNet_n_proc_DA=2 nnUNetv2_predict -d Dataset072_Prostate -i /path/to/imagesTs/ -o /path/to/output -f 0 1 2 3 4 -tr nnUNetTrainer_1000epochs -c 3d_fullres -p nnUNetPlans

```
## Postprocessing

- Once inference is completed, run postprocessing like this:

```
nnUNetv2_apply_postprocessing -i OUTPUT_FOLDER -o OUTPUT_FOLDER_PP -pp_pkl_file /scratch/nnUNet_results/Dataset072_Prostate/nnUNetTrainer_1000epochs__nnUNetPlans__3d_fullres_my_plan/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -np 8 -plans_json /scratch/nnUNet_results/Dataset072_Prostate/nnUNetTrainer_1000epochs__nnUNetPlans__3d_fullres_my_plan/crossval_results_folds_0_1_2_3_4/plans.json

```

- We therefore decided to implement a post-processing strategy previously validated by our team, which enables interpolation of the urethral segments along their centerline.

```

python inference/center_line_urethra_v1.py 

```

## Evaluation

The scripts used for evaluation are located in the evaluation folder.

- Metrics included:
  - Dice Similarity Coefficient
  - 95th percentile Hausdorff Distance (HD95) measured with MedPy
  - Mean Surface Distance
  - Volumetric overlap error
  - Relative volume difference

Example command:
```

python evaluation/calculs_metriques_prostatex.py 

```

## Pre-trained Models

- Pre-trained model weights are available in this repository under the /models directory.
- Place weights in:

```

\$NNUNET_RESULTS/Dataset072_Prostate/nnUNetTrainer_1000epochs__nnUNetPlans__3d_fullres_my_plan/fold_all/

```

- Optional: Convert predicted NIfTI masks to DICOM RTStruct for radiotherapy systems:

```

python inference/convert_prediction_to_rtstruct.py --input /path/to/output --dicom_series /path/to/reference_dicom --output /path/to/rtstruct/

```

## How to Reproduce

1. Prepare and convert datasets
2. Run planning and preprocessing
3. Train model folds
4. Run inference with ensemble
5. Run interpolation of the urethral segments along their centerline
6. Evaluate quantitative metrics

<div style="background-color:#ffdddd; border-left:6px solid #f44336; padding:10px;">
<strong>⚠️ Warning:</strong>  
Please make sure to check all folder and subfolder paths in the scripts.  
You must adapt these paths according to your own environment before running any code.
</div>


## How to cite

This work is currently under review; citation details will be updated after publication.

Please also cite the original nnU-Net paper:

```

@article{Isensee2021nnuNet,
title={nnU-Net: Self-adapting Framework for U-Net-based Medical Image Segmentation},
author={Isensee et al.},
journal={Nature Methods},
year={2021}
}

```

## Limitations and Use

- Intended for research use on multicentric prostate MRI for radiotherapy planning.
- Validated on specific MRI systems and acquisition protocols.
- Manual expert review mandatory before clinical application.
- Performance outside the training data domain (different MRI systems, protocols, pathologies) is not guaranteed.

## License

This project is licensed under the MIT License. It incorporates code from nnU-Net, which is licensed under Apache 2.0. Both licenses apply and must be respected. See LICENSE file for details.

## Contributing

Contributions, bug reports, and feature requests are welcome. Please open an issue or pull request.

Contact: miguel.castro@univ-rennes.fr
```
